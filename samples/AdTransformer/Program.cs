using Feather.NN;

const int sampleCount = 1024;
const int steps = 320;
const float learningRate = 0.03f;

Console.WriteLine("=== Feather NN Self-Attention Training ===");
Console.WriteLine($"samples={sampleCount} steps={steps}");

var (features, labels) = CreateSyntheticAttentionData(sampleCount, seed: 42);
using var model = new SelfAttentionClassifier(seed: 123);
using var optimizer = new Adam(model.Parameters, learningRate: learningRate, beta1: 0.85f, beta2: 0.99f, epsilon: 1e-6f);
using var trainer = model.CreateTrainer(features, labels, optimizer);

var initialLoss = trainer.EvaluateLoss();
Console.WriteLine($"initial loss={initialLoss:F6} accuracy={Accuracy(model, features, labels):F1}%");

var lastLoss = initialLoss;
for (var step = 0; step < steps; step++)
{
    lastLoss = trainer.TrainStep();

    if (step % 40 == 0 || step == steps - 1)
    {
        Console.WriteLine($"step {step + 1,4}/{steps}: loss={lastLoss:F6} accuracy={Accuracy(model, features, labels):F1}%");
    }
}

Console.WriteLine();
Console.WriteLine("final weights:");
Console.WriteLine(string.Join(", ", model.Weights.Value.Buffer.ToArray().Take(8).Select(v => v.ToString("F4"))));
Console.WriteLine($"final loss={lastLoss:F6} accuracy={Accuracy(model, features, labels):F1}%");
Console.WriteLine("inference:");
foreach (var (x0, x1) in new[] { (1.0f, 0.8f), (1.0f, -0.8f), (-1.0f, -0.8f), (-1.0f, 0.8f) })
{
    var score = model.PredictHost(x0, x1);
    Console.WriteLine($"  x=({x0,4:F1}, {x1,4:F1}) score={score:F4} predicted={(score > 0.5f ? 1 : 0)}");
}

Console.WriteLine($"dispatch={trainer.LastDispatchPath}, gradients materialized={trainer.GradientsMaterialized}");

static (float[] Features, float[] Labels) CreateSyntheticAttentionData(int count, int seed)
{
    var rng = new Random(seed);
    var features = new float[count * 2];
    var labels = new float[count];
    for (var i = 0; i < count; i++)
    {
        var x0 = NextGaussian(rng);
        var x1 = NextGaussian(rng);
        features[(i * 2) + 0] = x0;
        features[(i * 2) + 1] = x1;
        labels[i] = x0 * x1 > 0f ? 1f : 0f;
    }

    return (features, labels);
}

static float Accuracy(SelfAttentionClassifier model, float[] features, float[] labels)
{
    var correct = 0;
    for (var i = 0; i < labels.Length; i++)
    {
        var output = model.PredictHost(features[(i * 2) + 0], features[(i * 2) + 1]);
        var predicted = output > 0.5f ? 1f : 0f;
        if (predicted == labels[i])
        {
            correct++;
        }
    }

    return 100f * correct / labels.Length;
}

static float NextGaussian(Random rng)
{
    var u1 = 1.0 - rng.NextDouble();
    var u2 = 1.0 - rng.NextDouble();
    return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
}
