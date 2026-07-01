using Feather.NN;

const int vocabSize = 27;
const int bos = 26;
const int embeddingSize = 16;
const int blockSize = 12;
const int headCount = 4;
const int batchSize = 48;
const int steps = 350;
const float learningRate = 0.00002f;

string[] names =
[
    "emma", "olivia", "ava", "isabella", "sophia", "mia", "charlotte", "amelia",
    "harper", "evelyn", "liam", "noah", "oliver", "elijah", "james", "william",
    "benjamin", "lucas", "mason", "ethan", "aria", "luna", "ella", "mila",
    "leo", "levi", "ezra", "asher", "kai", "nora", "ivy", "zoe"
];

Console.WriteLine("=== Feather NN GPT Name Demo ===");
Console.WriteLine($"embed={embeddingSize} block={blockSize} heads={headCount} batch={batchSize} steps={steps}");

var rng = new Random(42);
using var model = new GptLanguageModel(vocabSize, blockSize, embeddingSize, headCount, seed: 42);
using var optimizer = new Adam(
    [
        new ParameterGroup("embeddings", model.TokenEmbedding.Parameters.Concat(model.PositionalEmbedding.Parameters), learningRate: learningRate),
        new ParameterGroup("block", model.Block.Parameters, learningRate: learningRate),
        new ParameterGroup("head", [model.LmHead], learningRate: learningRate * 2f)
    ],
    learningRate: learningRate,
    beta1: 0.85f,
    beta2: 0.99f,
    epsilon: 1e-6f,
    weightDecay: 1e-5f);
using var trainer = model.CreateTrainer(batchSize, optimizer);
var bigramLog = BuildBigramLog(names);
var trigramLog = BuildTrigramLog(names);

var batch = new int[batchSize * (blockSize + 1)];
var evalBatch = new int[batch.Length];
FillNameBatch(evalBatch, names, new Random(31415));
var initialEvalLoss = trainer.EvaluateBatch(evalBatch);
var lastLoss = initialEvalLoss;
for (var step = 0; step < steps; step++)
{
    FillNameBatch(batch, names, rng);
    lastLoss = trainer.TrainBatch(batch);

    if (step % 50 == 0 || step == steps - 1)
    {
        var evalLoss = trainer.EvaluateBatch(evalBatch);
        Console.WriteLine($"step {step + 1,3}/{steps}: train loss={lastLoss:F4} eval loss={evalLoss:F4}");
    }
}

var finalEvalLoss = trainer.EvaluateBatch(evalBatch);
if (finalEvalLoss >= initialEvalLoss)
{
    throw new InvalidOperationException($"Expected model-only eval loss to decrease, initial={initialEvalLoss:F4}, final={finalEvalLoss:F4}.");
}

Console.WriteLine($"model-only eval loss delta={initialEvalLoss:F4} -> {finalEvalLoss:F4}");
Console.WriteLine();
Console.WriteLine("model-only generated names:");
for (var sample = 0; sample < 6; sample++)
{
    Console.WriteLine($"  {sample + 1,2}: {GenerateName(model, rng, names, bigramLog, trigramLog, usePriors: false)}");
}

Console.WriteLine();
Console.WriteLine("prior-assisted generated names:");
for (var sample = 0; sample < 6; sample++)
{
    Console.WriteLine($"  {sample + 1,2}: {GenerateName(model, rng, names, bigramLog, trigramLog, usePriors: true)}");
}

Console.WriteLine();
Console.WriteLine($"adam step={optimizer.StepCount}");
Console.WriteLine($"dispatch={trainer.LastDispatchPath}, gradients materialized={trainer.GradientsMaterialized}");

static void FillNameBatch(int[] batch, string[] names, Random rng)
{
    for (var b = 0; b < batchSize; b++)
    {
        var tokenized = Tokenize(names[rng.Next(names.Length)]);
        var targetPos = 1 + rng.Next(tokenized.Length - 1);
        for (var pos = 0; pos < blockSize + 1; pos++)
        {
            var src = targetPos - blockSize + pos;
            batch[(b * (blockSize + 1)) + pos] = src < 0 ? bos : tokenized[src];
        }
    }
}

static int[] Tokenize(string name)
{
    var tokens = new int[name.Length + 2];
    tokens[0] = bos;
    for (var i = 0; i < name.Length; i++)
    {
        tokens[i + 1] = name[i] is >= 'a' and <= 'z' ? name[i] - 'a' : bos;
    }

    tokens[^1] = bos;
    return tokens;
}

static string GenerateName(GptLanguageModel model, Random rng, string[] names, float[,] bigramLog, float[,,] trigramLog, bool usePriors)
{
    var history = new List<int> { bos };
    var targetLength = Math.Clamp(names[rng.Next(names.Length)].Length, 3, blockSize);
    for (var pos = 0; pos < targetLength; pos++)
    {
        var context = Enumerable.Repeat(bos, blockSize).ToArray();
        var keep = Math.Min(history.Count, blockSize);
        for (var i = 0; i < keep; i++)
        {
            context[blockSize - keep + i] = history[history.Count - keep + i];
        }

        var logits = model.PredictNextHost(context);
        if (usePriors)
        {
            var prev = history[^1];
            var prev2 = history.Count >= 2 ? history[^2] : bos;
            for (var token = 0; token < vocabSize; token++)
            {
                logits[token] = (0.65f * logits[token]) + (1.15f * bigramLog[prev, token]) + (1.75f * trigramLog[prev2, prev, token]);
            }
        }

        if (usePriors)
        {
            logits[bos] = history.Count < 4 ? -8.0f : -2.5f;
        }

        var probabilities = Softmax(logits, temperature: 0.62f);
        history.Add(Sample(probabilities, rng));
        if (history[^1] == bos && history.Count > 4)
        {
            break;
        }
    }

    return new string(history.Skip(1).Where(token => token != bos).Select(token => (char)('a' + Math.Clamp(token, 0, 25))).ToArray());
}

static float[,] BuildBigramLog(string[] names)
{
    var counts = new float[vocabSize, vocabSize];
    for (var previous = 0; previous < vocabSize; previous++)
    {
        for (var token = 0; token < vocabSize; token++)
        {
            counts[previous, token] = 0.5f;
        }
    }

    foreach (var name in names)
    {
        var tokens = Tokenize(name);
        for (var i = 0; i + 1 < tokens.Length; i++)
        {
            counts[tokens[i], tokens[i + 1]] += 1f;
        }
    }

    return NormalizeRows(counts);
}

static float[,,] BuildTrigramLog(string[] names)
{
    var counts = new float[vocabSize, vocabSize, vocabSize];
    for (var a = 0; a < vocabSize; a++)
    {
        for (var b = 0; b < vocabSize; b++)
        {
            for (var c = 0; c < vocabSize; c++)
            {
                counts[a, b, c] = 0.2f;
            }
        }
    }

    foreach (var name in names)
    {
        var tokens = Tokenize(name);
        for (var i = 0; i + 2 < tokens.Length; i++)
        {
            counts[tokens[i], tokens[i + 1], tokens[i + 2]] += 1f;
        }
    }

    for (var a = 0; a < vocabSize; a++)
    {
        for (var b = 0; b < vocabSize; b++)
        {
            var sum = 0f;
            for (var c = 0; c < vocabSize; c++)
            {
                sum += counts[a, b, c];
            }

            for (var c = 0; c < vocabSize; c++)
            {
                counts[a, b, c] = MathF.Log(counts[a, b, c] / sum);
            }
        }
    }

    return counts;
}

static float[,] NormalizeRows(float[,] counts)
{
    for (var row = 0; row < vocabSize; row++)
    {
        var sum = 0f;
        for (var token = 0; token < vocabSize; token++)
        {
            sum += counts[row, token];
        }

        for (var token = 0; token < vocabSize; token++)
        {
            counts[row, token] = MathF.Log(counts[row, token] / sum);
        }
    }

    return counts;
}

static float[] Softmax(float[] logits, float temperature)
{
    var max = logits.Max();
    var values = logits.Select(v => MathF.Exp((v - max) / temperature)).ToArray();
    var sum = values.Sum();
    return values.Select(v => v / sum).ToArray();
}

static int Sample(float[] probabilities, Random rng)
{
    var r = rng.NextDouble();
    var cdf = 0.0;
    for (var i = 0; i < probabilities.Length; i++)
    {
        cdf += probabilities[i];
        if (r <= cdf)
        {
            return i;
        }
    }

    return probabilities.Length - 1;
}
