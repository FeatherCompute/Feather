// Trains a packed 3->h->h->1 ReLU regression MLP on the GPU through the host-drivable
// ITrainingJob contract, then writes an atomic checkpoint the way a long-lived host would.
//
// The point of the sample is the shape of the loop, not the model: a host owns the step counter and
// the reporting cadence, the job owns the device state, and loss readback happens only on the steps
// that report. Everything here is what a Blender-side training panel drives.

using Feather;
using Feather.NN;
using Feather.Resources;

const int hiddenSize = 12;
const int sampleCount = 96;
const int steps = 600;
const int reportEvery = 50;
const float learningRate = 0.02f;

// Target: a smooth nonlinear function of three inputs, which a two-layer ReLU network has to bend to
// fit rather than being able to solve linearly.
static float Target(float x, float y, float z) => (0.6f * x * y) + MathF.Sin(1.5f * z) - (0.3f * x) + 0.2f;

var random = new Random(20240607);
var inputs = new float[sampleCount * MlpLayout.InputSize];
var targets = new float[sampleCount];
for (var i = 0; i < sampleCount; i++)
{
    var x = (float)((random.NextDouble() * 2.0) - 1.0);
    var y = (float)((random.NextDouble() * 2.0) - 1.0);
    var z = (float)((random.NextDouble() * 2.0) - 1.0);
    inputs[(i * MlpLayout.InputSize) + 0] = x;
    inputs[(i * MlpLayout.InputSize) + 1] = y;
    inputs[(i * MlpLayout.InputSize) + 2] = z;
    targets[i] = Target(x, y, z);
}

var checkpointDirectory = Path.Combine(Path.GetTempPath(), "feather-ad-mlp-regression");
Directory.CreateDirectory(checkpointDirectory);
const string checkpointName = "mlp-regression.fthc";

// The context is what a host owns: the project root checkpoints resolve against, a cancellation token
// wired to its stop button, and a loss sink feeding whatever UI it has.
var losses = new List<(int Step, float Loss)>();
using var cancellation = new CancellationTokenSource();
var context = new TrainingContext(
    checkpointDirectory,
    cancellation.Token,
    lossStream: report => losses.Add((report.Step, report.Loss)));

using var job = new MlpRegressionJob(hiddenSize, inputs, targets, learningRate) { PlannedSteps = steps };
job.Initialize(context);

Console.WriteLine("AD MLP Regression (3 -> {0} -> {0} -> 1)", hiddenSize);
Console.WriteLine($"samples={sampleCount}, packed weights={MlpLayout.PackedElementCount3To1(hiddenSize)}, steps={steps}, lr={learningRate}");
Console.WriteLine();
Console.WriteLine("step     loss");

var lastReport = TrainingStepReport.Unreported(0, DispatchPath.None);
for (var step = 0; step < steps; step++)
{
    if (context.CancellationToken.IsCancellationRequested)
    {
        break;
    }

    context.AdvanceTo(step);

    // A reporting step pays the loss readback; the rest do not. That split is why
    // RunWithoutLossReadback exists on the training-step path.
    var isReportingStep = step % reportEvery == 0 || step == steps - 1;
    var report = isReportingStep ? job.StepAndReadLoss(context) : job.Step(context);
    if (report.HasDiverged)
    {
        Console.WriteLine($"{step,4} diverged (loss={report.Loss})");
        break;
    }

    if (report.IsReported)
    {
        lastReport = report;
        context.ReportLoss(report);
        Console.WriteLine($"{step,4} {report.Loss,10:F6}");
    }
}

Console.WriteLine();
Console.WriteLine($"dispatch={lastReport.DispatchPath}");
Console.WriteLine($"first reported loss={losses[0].Loss:F6}, final={losses[^1].Loss:F6}, reduction={losses[0].Loss / MathF.Max(losses[^1].Loss, 1e-9f):F1}x");

// SaveAtomic writes a sibling temp file and replaces the target, so a host killed mid-save leaves the
// previous checkpoint intact rather than a truncated one.
var checkpointPath = context.ResolveProjectPath(checkpointName);
Checkpoint.SaveAtomic(
    checkpointPath,
    job.Parameters,
    new CheckpointMetadata(
        lastReport.Step,
        lastReport.Loss,
        ModelKind: "mlp-regression-3to1",
        Tags: new Dictionary<string, string> { ["hiddenSize"] = hiddenSize.ToString() }));
Console.WriteLine($"checkpoint written to {checkpointPath}");

// Inspect reads the header without allocating a single GPU buffer, which is how a host populates a
// checkpoint picker.
var info = Checkpoint.Inspect(checkpointPath);
Console.WriteLine($"inspect: version={info.Version}, weights={info.WeightCount}, savedAt={info.Metadata?.SavedAtUtc:u}, kind={info.Metadata?.ModelKind}");

// The inference side: load the checkpoint once and hold it, then bind its buffer view straight into
// the inference kernel. That binding is the whole handoff — nothing is copied back to the host and no
// weight is re-uploaded per frame.
const int probeCount = 6;
using var weights = InferenceWeights.Load(checkpointPath);
using var probeInputs = GPU.CreateBuffer<float>(inputs.AsSpan(0, probeCount * MlpLayout.InputSize));
using var probeScratch = GPU.CreateBuffer<float>(MlpLayout.ScratchElementsPerLane3To1(hiddenSize) * probeCount);
using var probePredictions = GPU.CreateBuffer<float>(probeCount);
GpuKernel.Dispatch(
    GPU.Context,
    new MlpInference3To1Kernel(
        probeInputs.AsReadOnly(),
        weights.Buffer(MlpRegressionJob.WeightParameterName),
        probeScratch.AsReadWrite(),
        probePredictions.AsReadWrite(),
        new Uniform<int>(hiddenSize)),
    new GpuDispatchSize(probeCount, 1, 1),
    wait: true);

var predicted = probePredictions.ToArray();
Console.WriteLine();
Console.WriteLine("       x        y        z   predicted     target");
for (var i = 0; i < probeCount; i++)
{
    Console.WriteLine(
        $"{inputs[(i * MlpLayout.InputSize) + 0],8:F4} {inputs[(i * MlpLayout.InputSize) + 1],8:F4} " +
        $"{inputs[(i * MlpLayout.InputSize) + 2],8:F4} {predicted[i],11:F5} {targets[i],10:F5}");
}
