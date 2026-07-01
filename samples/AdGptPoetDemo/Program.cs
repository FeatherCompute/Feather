using Feather.NN;
using System.Diagnostics;

const string vocabulary = "\n abcdefghijklmnopqrstuvwxyz',.;:?!-";
const int embeddingSize = 16;
const int blockSize = 32;
const int headCount = 4;
const int batchSize = 64;
const int steps = 1500;
const int logEvery = 250;
const float learningRate = 0.00008f;

const string corpus = """
shall i compare thee to a summer's day?
thou art more lovely and more temperate:
rough winds do shake the darling buds of may,
and summer's lease hath all too short a date:
sometime too hot the eye of heaven shines,
and often is his gold complexion dimm'd;
and every fair from fair sometime declines,
by chance or nature's changing course untrimm'd;
but thy eternal summer shall not fade,
nor lose possession of that fair thou ow'st;
nor shall death brag thou wander'st in his shade,
when in eternal lines to time thou grow'st:
so long as men can breathe or eyes can see,
so long lives this, and this gives life to thee.

how do i love thee? let me count the ways.
i love thee to the depth and breadth and height
my soul can reach, when feeling out of sight
for the ends of being and ideal grace.
i love thee to the level of every day's
most quiet need, by sun and candle-light.
i love thee freely, as men strive for right;
i love thee purely, as they turn from praise.

let me not to the marriage of true minds
admit impediments.
love is not love
which alters when it alteration finds,
or bends with the remover to remove:
o no!
it is an ever-fixed mark,
that looks on tempests and is never shaken;
it is the star to every wandering bark,
whose worth's unknown, although his height be taken.
love's not time's fool, though rosy lips and cheeks
within his bending sickle's compass come:
love alters not with his brief hours and weeks,
but bears it out even to the edge of doom.
if this be error and upon me proved,
i never writ, nor no man ever loved.

how do i love thee?
let me count the ways.
i love thee to the depth and breadth and height
my soul can reach, when feeling out of sight
for the ends of being and ideal grace.
i love thee to the level of every day's
most quiet need, by sun and candle-light.
i love thee freely, as men strive for right.
i love thee purely, as they turn from praise.
i love thee with the passion put to use
in my old griefs, and with my childhood's faith.
i love thee with a love i seemed to lose
with my lost saints,
i love thee with the breath,
smiles, tears, of all my life!
and, if god choose,
i shall but love thee better after death.

she walks in beauty, like the night
of cloudless climes and starry skies;
and all that's best of dark and bright
meet in her aspect and her eyes:
thus mellowed to that tender light
which heaven to gaudy day denies.
one shade the more, one ray the less,
had half impaired the nameless grace
which waves in every raven tress,
or softly lightens o'er her face;
where thoughts serenely sweet express,
how pure, how dear their dwelling-place.
and on that cheek, and o'er that brow,
so soft, so calm, yet eloquent,
the smiles that win, the tints that glow,
but tell of days in goodness spent,
a mind at peace with all below,
a heart whose love is innocent!

when you are old and grey and full of sleep,
and nodding by the fire, take down this book,
and slowly read, and dream of the soft look
your eyes had once, and of their shadows deep;
how many loved your moments of glad grace,
and loved your beauty with love false or true,
but one man loved the pilgrim soul in you,
and loved the sorrows of your changing face;
and bending down beside the glowing bars,
murmur, a little sadly, how love fled
and paced upon the mountains overhead
and hid his face amid a crowd of stars.

bright star, would i were steadfast as thou art-
not in lone splendour hung aloft the night
and watching, with eternal lids apart,
like nature's patient, sleepless eremite,
the moving waters at their priestlike task
of pure ablution round earth's human shores,
or gazing on the new soft-fallen mask
of snow upon the mountains and the moors-
no-yet still stedfast, still unchangeable,
pillow'd upon my fair love's ripening breast,
to feel for ever its soft fall and swell,
awake for ever in a sweet unrest,
still, still to hear her tender-taken breath,
and so live ever-or else swoon to death.

i loved you first: but afterwards your love
outsoaring mine, sang such a loftier song
as drowned the friendly cooings of my dove.
which owes the other most?
my love was long,
and yet one day you grew above my heart,
and gave it deeper love than it could hold.
love is not love that waits for love's return,
but love that gives, not counting cost or gold.

come live with me and be my love,
and we will all the pleasures prove
that valleys, groves, hills, and fields,
woods, or steepy mountain yields.
and we will sit upon the rocks,
seeing the shepherds feed their flocks,
by shallow rivers to whose falls
melodious birds sing madrigals.
and i will make thee beds of roses
and a thousand fragrant posies,
a cap of flowers, and a kirtle
embroidered all with leaves of myrtle;
a gown made of the finest wool
which from our pretty lambs we pull;
fair lined slippers for the cold,
with buckles of the purest gold;
a belt of straw and ivy buds,
with coral clasps and amber studs.
and if these pleasures may thee move,
come live with me and be my love.

how sweet i roamed from field to field,
and tasted all the summer's pride,
till i the prince of love beheld,
who in the sunny beams did glide!
he showed me lilies for my hair,
and blushing roses for my brow;
he led me through his gardens fair
where all his golden pleasures grow.
with sweet may dews my wings were wet,
and phoebus fired my vocal rage;
he caught me in his silken net,
and shut me in his golden cage.
he loves to sit and hear me sing,
then, laughing, sports and plays with me;
then stretches out my golden wing,
and mocks my loss of liberty.

love is a smoke raised with the fume of sighs;
being purged, a fire sparkling in lovers' eyes;
being vexed, a sea nourished with lovers' tears.
what is it else?
a madness most discreet,
a choking gall and a preserving sweet.

farewell, thou art too dear for my possessing,
and like enough thou know'st thy estimate:
the charter of thy worth gives thee releasing;
my bonds in thee are all determinate.
for how do i hold thee but by thy granting?
and for that riches where is my deserving?
the cause of this fair gift in me is wanting,
and so my patent back again is swerving.
thyself thou gavest, thy own worth then not knowing,
or me, to whom thou gavest it, else mistaking;
so thy great gift, upon misprision growing,
comes home again, on better judgement making.
thus have i had thee, as a dream doth flatter,
in sleep a king, but waking no such matter.

love is not love
which alters when it alteration finds,
or bends with the remover to remove:
o no! it is an ever-fixed mark,
that looks on tempests and is never shaken.

love seeketh not itself to please,
nor for itself hath any care,
but for another gives its ease,
and builds a heaven in hell's despair.
""";

Console.WriteLine("=== Feather NN GPT Poet Demo ===");
Console.WriteLine($"vocab={vocabulary.Length} embed={embeddingSize} block={blockSize} batch={batchSize} steps={steps}");

var text = Normalize(corpus);
var ids = Encode(text);
var rng = new Random(42);
using var model = new GptLanguageModel(vocabulary.Length, blockSize, embeddingSize, headCount, seed: 42);
var priors = BuildPriors(ids);
using var optimizer = new Adam(
    [
        new ParameterGroup("embeddings", model.TokenEmbedding.Parameters.Concat(model.PositionalEmbedding.Parameters), learningRate: learningRate),
        new ParameterGroup("block", model.Block.Parameters, learningRate: learningRate),
        new ParameterGroup("head", [model.LmHead], learningRate: learningRate)
    ],
    learningRate: learningRate,
    beta1: 0.85f,
    beta2: 0.99f,
    epsilon: 1e-5f,
    weightDecay: 1e-4f,
    gradientClip: 0.25f);
using var trainer = model.CreateTrainer(batchSize, optimizer);

var batch = new int[batchSize * (blockSize + 1)];
var evalBatch = new int[batch.Length];
FillSequenceBatch(evalBatch, ids, new Random(31415));
var initialEvalLoss = trainer.EvaluateBatch(evalBatch);
var lastLoss = initialEvalLoss;
var lastEvalLoss = initialEvalLoss;
var bestEvalLoss = initialEvalLoss;
var bestStep = 0;
var stoppedEarly = false;
var stopReason = string.Empty;
var trainingStopwatch = Stopwatch.StartNew();
for (var step = 0; step < steps; step++)
{
    FillSequenceBatch(batch, ids, rng);
    lastLoss = trainer.TrainBatch(batch);

    if (step == 0)
    {
        var firstStepGradients = SummarizeGradients(model);
        Console.WriteLine(
            $"first-step reduced grad avgAbs={firstStepGradients.AverageAbs:F6} nonzero={firstStepGradients.NonZero}/{firstStepGradients.Count}");
    }

    if (step % logEvery == 0 || step == steps - 1)
    {
        var evalLoss = trainer.EvaluateBatch(evalBatch);
        lastEvalLoss = evalLoss;
        if (evalLoss < bestEvalLoss)
        {
            bestEvalLoss = evalLoss;
            bestStep = step + 1;
        }

        Console.WriteLine(
            $"step {step + 1,4}/{steps}: raw train={lastLoss:F4} raw eval={evalLoss:F4} best={bestEvalLoss:F4}@{bestStep} | adam step {optimizer.StepCount}");

        if (!float.IsFinite(lastLoss) || !float.IsFinite(evalLoss))
        {
            stoppedEarly = true;
            stopReason = $"non-finite loss at step {step + 1}";
            break;
        }

        if (step + 1 > 100 && evalLoss > MathF.Max(initialEvalLoss + 1.0f, bestEvalLoss + 1.5f))
        {
            stoppedEarly = true;
            stopReason = $"raw eval divergence at step {step + 1}: eval={evalLoss:F4}, best={bestEvalLoss:F4}";
            break;
        }
    }
}

trainingStopwatch.Stop();
var finalEvalLoss = trainer.EvaluateBatch(evalBatch);
if (stoppedEarly)
{
    throw new InvalidOperationException($"Poet training stopped early because {stopReason}. Raw final eval={finalEvalLoss:F4}; best observed={bestEvalLoss:F4}@{bestStep}.");
}

if (finalEvalLoss >= initialEvalLoss)
{
    throw new InvalidOperationException($"Expected model-only eval loss to decrease for the raw GPT path, initial={initialEvalLoss:F4}, final={finalEvalLoss:F4}, best={bestEvalLoss:F4}@{bestStep}.");
}

var uniformLoss = MathF.Log(vocabulary.Length);
if (finalEvalLoss >= uniformLoss - 0.20f)
{
    throw new InvalidOperationException($"Expected raw model-only eval loss to beat the uniform character baseline by at least 0.20 nats, uniform={uniformLoss:F4}, final={finalEvalLoss:F4}.");
}

if (finalEvalLoss >= initialEvalLoss - 0.50f)
{
    throw new InvalidOperationException($"Expected raw model-only eval loss to improve by at least 0.50 nats, initial={initialEvalLoss:F4}, final={finalEvalLoss:F4}, best={bestEvalLoss:F4}@{bestStep}.");
}

Console.WriteLine($"model-only eval loss delta={initialEvalLoss:F4} -> {finalEvalLoss:F4}; best observed={bestEvalLoss:F4}@{bestStep}; last periodic eval={lastEvalLoss:F4}");
Console.WriteLine($"training wall time={trainingStopwatch.Elapsed.TotalSeconds:F2}s");
Console.WriteLine();
const string modelOnlyPrompt = "shall i ";
var modelOnlySample = Generate(model, priors, modelOnlyPrompt, 260, new Random(1701), usePriors: false);
var priorAssistedSample = Generate(model, priors, "shall i ", 900, new Random(42), usePriors: true);
Console.WriteLine("model-only generated poem:");
Console.WriteLine(modelOnlySample);
Console.WriteLine();
Console.WriteLine("prior-assisted generated poem:");
Console.WriteLine(priorAssistedSample);
Console.WriteLine();
Console.WriteLine($"adam step={optimizer.StepCount}");
Console.WriteLine($"dispatch={trainer.LastDispatchPath}, gradients materialized={trainer.GradientsMaterialized}");

static void FillSequenceBatch(int[] batch, int[] ids, Random rng)
{
    for (var row = 0; row < batchSize; row++)
    {
        var start = rng.Next(0, ids.Length - blockSize - 1);
        for (var i = 0; i < blockSize + 1; i++)
        {
            batch[(row * (blockSize + 1)) + i] = ids[start + i];
        }
    }
}

static string Generate(GptLanguageModel model, PoetryPriors priors, string prompt, int count, Random rng, bool usePriors)
{
    var history = Encode(Normalize(prompt)).ToList();
    for (var i = 0; i < count; i++)
    {
        var context = Enumerable.Repeat(CharToIndex('\n'), blockSize).ToArray();
        var keep = Math.Min(history.Count, blockSize);
        for (var j = 0; j < keep; j++)
        {
            context[blockSize - keep + j] = history[history.Count - keep + j];
        }

        var logits = model.PredictNextHost(context);
        if (usePriors)
        {
            BlendPoetryPriors(logits, history, priors);
            BiasForReadableLines(logits, history);
        }

        var probabilities = usePriors
            ? Softmax(logits, temperature: 0.70f)
            : TopKSoftmax(logits, temperature: 0.40f, topK: 8);
        history.Add(Sample(probabilities, rng));
    }

    return new string(history.Select(IndexToChar).ToArray());
}

static (double AverageAbs, int NonZero, int Count) SummarizeGradients(GptLanguageModel model)
{
    double averageAbs = 0;
    var nonZero = 0;
    var count = 0;
    foreach (var parameter in model.Parameters.OfType<Parameter<float>>())
    {
        double sumAbs = 0;
        var length = 0;
        foreach (var gradient in parameter.Gradient.Buffer.ToArray())
        {
            sumAbs += Math.Abs(gradient);
            length++;
        }

        var meanAbs = length == 0 ? 0 : sumAbs / length;
        averageAbs += meanAbs;
        if (meanAbs > 1e-8)
        {
            nonZero++;
        }

        count++;
    }

    return count == 0 ? (0, 0, 0) : (averageAbs / count, nonZero, count);
}

static void BlendPoetryPriors(float[] logits, List<int> history, PoetryPriors priors)
{
    var prev = history.Count >= 1 ? history[^1] : CharToIndex('\n');
    var prev2 = history.Count >= 2 ? history[^2] : CharToIndex('\n');
    var prev3 = history.Count >= 3 ? history[^3] : CharToIndex('\n');
    var prev4 = history.Count >= 4 ? history[^4] : CharToIndex('\n');
    var triRow = (prev2 * vocabulary.Length) + prev;
    var fourRow = ((prev3 * vocabulary.Length) + prev2) * vocabulary.Length + prev;
    var fiveKey = (((prev4 * vocabulary.Length) + prev3) * vocabulary.Length + prev2) * vocabulary.Length + prev;
    var gptProbabilities = Softmax(logits, temperature: 1.0f);
    for (var token = 0; token < vocabulary.Length; token++)
    {
        logits[token] =
            (0.35f * MathF.Log(MathF.Max(gptProbabilities[token], 1e-8f))) +
            (0.10f * priors.BigramLog[prev, token]) +
            (0.35f * priors.TrigramLog[triRow, token]) +
            (0.75f * priors.FourgramLog[fourRow, token]);
    }

    if (priors.FivegramLog.TryGetValue(fiveKey, out var fivegram))
    {
        for (var token = 0; token < vocabulary.Length; token++)
        {
            logits[token] += 2.80f * fivegram[token];
        }
    }

    var key = ContextKey(history, 8);
    if (key is not null && priors.LongContextLog.TryGetValue(key, out var longContext))
    {
        for (var token = 0; token < vocabulary.Length; token++)
        {
            logits[token] += 3.50f * longContext[token];
        }
    }

    ApplyRecentRepeatPenalty(logits, history);
}

static void BiasForReadableLines(float[] logits, List<int> history)
{
    var newline = CharToIndex('\n');
    var lineLength = 0;
    for (var i = history.Count - 1; i >= 0 && IndexToChar(history[i]) != '\n'; i--)
    {
        lineLength++;
    }

    if (lineLength < 22)
    {
        logits[newline] -= 3.0f;
    }

    if (lineLength > 52)
    {
        logits[newline] += 2.5f;
    }

    if (lineLength > 68)
    {
        logits[newline] += 5.0f;
    }
}

static void ApplyRecentRepeatPenalty(float[] logits, List<int> history)
{
    if (history.Count < 3)
    {
        return;
    }

    var prev = history[^1];
    var prev2 = history[^2];
    var prev3 = history[^3];
    var recentStart = Math.Max(0, history.Count - 180);
    for (var token = 0; token < vocabulary.Length; token++)
    {
        var repeats = 0;
        for (var i = recentStart; i + 3 < history.Count; i++)
        {
            if (history[i] == prev3 &&
                history[i + 1] == prev2 &&
                history[i + 2] == prev &&
                history[i + 3] == token)
            {
                repeats++;
            }
        }

        logits[token] -= 0.9f * repeats;
    }
}

static PoetryPriors BuildPriors(int[] ids)
{
    var vocab = vocabulary.Length;
    var bigram = new float[vocab, vocab];
    var trigram = new float[vocab * vocab, vocab];
    var fourgram = new float[vocab * vocab * vocab, vocab];
    var fivegram = new Dictionary<int, float[]>();
    for (var a = 0; a < vocab; a++)
    {
        for (var b = 0; b < vocab; b++)
        {
            bigram[a, b] = 0.5f;
        }
    }

    for (var row = 0; row < vocab * vocab; row++)
    {
        for (var token = 0; token < vocab; token++)
        {
            trigram[row, token] = 0.12f;
        }
    }

    for (var row = 0; row < vocab * vocab * vocab; row++)
    {
        for (var token = 0; token < vocab; token++)
        {
            fourgram[row, token] = 0.03f;
        }
    }

    for (var i = 0; i + 1 < ids.Length; i++)
    {
        bigram[ids[i], ids[i + 1]] += 1f;
    }

    for (var i = 0; i + 2 < ids.Length; i++)
    {
        trigram[(ids[i] * vocab) + ids[i + 1], ids[i + 2]] += 1f;
    }

    for (var i = 0; i + 3 < ids.Length; i++)
    {
        fourgram[((ids[i] * vocab) + ids[i + 1]) * vocab + ids[i + 2], ids[i + 3]] += 1f;
    }

    for (var i = 0; i + 4 < ids.Length; i++)
    {
        var key = (((ids[i] * vocab) + ids[i + 1]) * vocab + ids[i + 2]) * vocab + ids[i + 3];
        if (!fivegram.TryGetValue(key, out var counts))
        {
            counts = Enumerable.Repeat(0.01f, vocab).ToArray();
            fivegram.Add(key, counts);
        }

        counts[ids[i + 4]] += 1f;
    }

    var longContext = new Dictionary<string, float[]>(StringComparer.Ordinal);
    for (var i = 0; i + 8 < ids.Length; i++)
    {
        var key = new string(ids.Skip(i).Take(8).Select(id => (char)id).ToArray());
        if (!longContext.TryGetValue(key, out var counts))
        {
            counts = Enumerable.Repeat(0.01f, vocab).ToArray();
            longContext.Add(key, counts);
        }

        counts[ids[i + 8]] += 1f;
    }

    NormalizeRows(bigram);
    NormalizeRows(trigram);
    NormalizeRows(fourgram);
    foreach (var counts in fivegram.Values)
    {
        NormalizeRow(counts);
    }

    foreach (var counts in longContext.Values)
    {
        NormalizeRow(counts);
    }

    return new PoetryPriors(bigram, trigram, fourgram, fivegram, longContext);
}

static void NormalizeRows(float[,] rows)
{
    var rowCount = rows.GetLength(0);
    var columns = rows.GetLength(1);
    for (var row = 0; row < rowCount; row++)
    {
        var values = new float[columns];
        for (var column = 0; column < columns; column++)
        {
            values[column] = rows[row, column];
        }

        NormalizeRow(values);
        for (var column = 0; column < columns; column++)
        {
            rows[row, column] = values[column];
        }
    }
}

static void NormalizeRow(float[] values)
{
    var sum = values.Sum();
    for (var i = 0; i < values.Length; i++)
    {
        values[i] = MathF.Log(values[i] / sum);
    }
}

static string? ContextKey(List<int> history, int length)
{
    if (history.Count < length)
    {
        return null;
    }

    return new string(history.Skip(history.Count - length).Take(length).Select(id => (char)id).ToArray());
}

static string Normalize(string text)
{
    var chars = new List<char>(text.Length);
    var lastSpace = false;
    foreach (var raw in text)
    {
        var c = char.ToLowerInvariant(raw);
        if (!vocabulary.Contains(c))
        {
            c = ' ';
        }

        if (c == ' ' && lastSpace)
        {
            continue;
        }

        chars.Add(c);
        lastSpace = c == ' ';
    }

    return new string(chars.ToArray());
}

static int[] Encode(string text)
{
    var ids = new int[text.Length];
    for (var i = 0; i < text.Length; i++)
    {
        ids[i] = CharToIndex(text[i]);
    }

    return ids;
}

static int CharToIndex(char c)
{
    var index = vocabulary.IndexOf(c);
    return index >= 0 ? index : vocabulary.IndexOf(' ');
}

static char IndexToChar(int index)
    => index >= 0 && index < vocabulary.Length ? vocabulary[index] : '?';

static float[] Softmax(float[] logits, float temperature)
{
    var max = logits.Max();
    var values = logits.Select(v => MathF.Exp((v - max) / temperature)).ToArray();
    var sum = values.Sum();
    return values.Select(v => v / sum).ToArray();
}

static float[] TopKSoftmax(float[] logits, float temperature, int topK)
{
    var threshold = logits
        .OrderByDescending(static value => value)
        .Take(Math.Clamp(topK, 1, logits.Length))
        .Last();
    var filtered = new float[logits.Length];
    for (var i = 0; i < logits.Length; i++)
    {
        filtered[i] = logits[i] >= threshold ? logits[i] : -1e9f;
    }

    return Softmax(filtered, temperature);
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

internal sealed record PoetryPriors(
    float[,] BigramLog,
    float[,] TrigramLog,
    float[,] FourgramLog,
    Dictionary<int, float[]> FivegramLog,
    Dictionary<string, float[]> LongContextLog);
