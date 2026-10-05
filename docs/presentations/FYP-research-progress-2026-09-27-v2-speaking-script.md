# Speaking script — Active learning for historical OCR

For the [18-slide research presentation](FYP-research-progress-2026-09-27-v2-final.pptx). Approximate delivery time: 12–14 minutes at a conversational pace. The slide numbers below match the PowerPoint and PDF. This is meant to be spoken, so the detailed tables remain on screen rather than being read row by row.

## 1. Research question

My final-year project asks whether we can choose *which historical pages to label* more intelligently, so that a vision-language OCR model learns from fewer labelled pages. There are two distinct questions here: can the model learn to transcribe this handwriting at all, and does an active-learning strategy choose better pages than random sampling? I first had to establish a reliable recognition task before the second question could be answered. I will show that progression, the evidence from EXP-009 to EXP-013, and where the claims stop.

## 2. Dataset and annotation unit

I use READ2016’s Bozen historical German handwriting. It has 350 official training pages and 50 validation pages. Eight validation pages were used for recipe checks; the reported learning curves use the other fixed 42. In this research, “annotation” is simulated: when I select a training page, I reveal its existing true line transcripts. No person is asked to label the image during these runs. The selector sees candidate images, not their hidden transcripts, and validation truth is reserved for evaluation. A *page*, rather than a line, is the unit acquired.

## 3. The active-learning loop

The loop begins with a zero-page evaluation of the untouched original model. Then a selection method chooses pages from the remaining training pool. Only after selection do I reveal those pages’ true transcripts. Each round fits a new LoRA adapter from the original base model on *all* labels acquired so far; it does not keep training the previous round’s adapter. I evaluate every checkpoint on the same validation slice and record the selections and model identity. Each experiment stops after three rounds fixed in advance, rather than stopping when a validation result happens to look good.

## 4. Why the original joint-OCR recipe was paused

The earliest experiments asked Qwen to produce both line boxes and their text from a whole page. I eventually made the training, checkpoint reload and evaluation path work, and tried stronger fitting and simpler structured outputs. But the OCR predictions remained invalid, repetitive or cut off; character error stayed near one and matched line-box F1 at zero. That is an important methodological result: if the basic output is unusable, differences between page-selection methods are hard to interpret. I therefore paused this *tested recipe* and isolated transcription first. It does not mean that detecting boxes and text jointly is impossible.

## 5. Why LightOnOCR and Qwen

EXP-009 and EXP-010 asked a simpler question: can historical **whole-page text** be learned when the model outputs transcription rather than box coordinates? I chose LightOnOCR-2-1B-base as a compact OCR-focused specialist designed for page transcription and domain adaptation. I chose Qwen3-VL-4B-Instruct as a small general vision-language model with document-OCR capability, and to test whether the earlier Qwen difficulty was partly the joint output task. Both used the same numeric page budgets and fine-tuning schedule. Their processors and prompts differ, so this is a practical control comparison, not a claim that one architecture is intrinsically superior.

## 6. EXP-009 and EXP-010: whole-page learning controls

Both models first predicted with **zero** training pages, then followed random selections of 64, 128 and 192 pages. On the same 42-page validation slice, LightOnOCR’s character error rate fell from 0.781 to 0.216; Qwen’s fell from 0.790 to 0.211. Word error also fell, to 0.618 and 0.559 respectively, so transcription improved but remained imperfect. Qwen’s untouched model hit the generation limit on ten of 42 pages, while its final checkpoint hit none; that matters when interpreting its starting score. These are **random-selection controls**. They demonstrate adaptation to the corpus, not an advantage from active learning.

## 7. Random, entropy and least confidence

To compare selection methods, I kept random sampling as the essential control. The first model-guided method is **entropy**: it averages uncertainty in the next-token distributions along generated text and selects pages with the highest score. The second is **least confidence**: it takes a length-normalized probability of the generated transcription and selects pages with the lowest score. The formulas on this slide show the token normalization. These are established uncertainty ideas adapted to generated OCR, rather than new strategies I claim to have invented. They are proxies for useful labels; a model can be confidently wrong. EXP-011 scores whole-page outputs, while EXP-013 averages scores from three fixed line crops per page.

## 8. K-center and visual diversity

K-center asks a different question: which page adds the most *visual coverage*? I represent each page with frozen visual features from the original model. After some pages have been selected, the algorithm greedily picks the remaining page farthest from its nearest selected page, then updates those distances. This follows the core-set intuition in earlier active-learning research. In EXP-011 it begins from the shared 64-page set. In EXP-013, with no selected pages, it starts at the page nearest the feature mean and then expands coverage. Visual diversity is a useful comparison, but it may not match differences in transcription difficulty.

## 9. EXP-011: what helps after a shared warm start?

EXP-011 was the first direct whole-page selection comparison. Within each of three seeds, all four methods shared the *same initial 64 randomly selected pages and checkpoint*. They then chose additional pages to reach 128 and 192. That design asks whether the **next** pages chosen by entropy, least confidence or k-center help more than additional random pages. The table shows means across the three paired seeds. At 128 pages, entropy’s mean character error was about 0.275 versus 0.286 for random. At 192, both were about 0.210. The promising early difference did not persist clearly. Seed 824’s untouched-model baseline was reused from EXP-010 with its original receipt, not rerun as a new result.

## 10. EXP-011: why one promising seed was not enough

Here the difference is always *strategy minus random within the same seed*, so a negative value favours the strategy. Entropy improved 128-page character error by about 1.03 percentage points on average, but at 192 pages its mean character-error difference was effectively zero. The standard deviations and word-error differences show that the ranking is not stable. I also found a concrete weakness of whole-page uncertainty scoring: in one pool pass, 26 outputs hit the token limit yet showed low entropy and high confidence. Repetitive text can look certain to the model. Output cutoffs therefore affect small comparisons and motivated the move to shorter, known-box transcription. The full validation scores remain the primary results.

## 11. Why use the supplied line boxes?

READ2016 already gives the location and reading order of its text lines. In the known-box pipeline, I crop each supplied region, ask Qwen to transcribe **text only**, and put the results back in source reading order. The image and crop shown here are actual dataset inputs. This removes line localization from the measured task and shortens each generated answer, making recognition and selection easier to study. It is **recognition with known locations, not box detection**. Because the task, input and training recipe differ from whole-page OCR, I do not treat a lower known-box error as proof that cropping alone caused an improvement.

## 12. EXP-012: establish the known-box random baseline

Before testing selectors on these crops, EXP-012 checked that the pipeline could learn with ordinary random selection. It evaluated the untouched Qwen model and then trained on 64, 128 and 192 selected pages. Across the same 861 validation line regions, character error fell from 0.804 to 0.190 after the first 64 pages, then to 0.123 and 0.094. Word error reached 0.370. The first acquisition produced the largest reduction, so smaller budgets might reveal meaningful selection differences earlier. But EXP-012 contains **only random sampling**: it establishes a working control curve, not active-learning benefit.

## 13. EXP-013: strategy-first, smaller budgets

EXP-013 therefore tests 0, 24, 48 and 72 acquired pages. Unlike EXP-011, there is **no shared random training checkpoint**: each method selects its own first 24 pages. A model-based method scores three predetermined line positions on each candidate page, but selecting the page reveals *all* its line transcripts for training. Entropy and confidence use the untouched model in the first round, then their own preceding adapters; k-center keeps its visual features frozen. The zero-page result is explicitly reused from EXP-012 after matching the model, processing, validation slice and evaluator, rather than presented as a fresh run.

## 14. EXP-013: read the learning curves

At 24 pages, entropy’s character error was 0.356 against 0.410 for random, a 5.45-percentage-point improvement; least confidence also beat random. But as the budget grew to 48 and 72 pages, random had the lowest character **and** word error. The curves converge rather than showing a sustained advantage. This is the key distinction between *the model learning from more labels*, which all four curves show, and *one selection rule consistently learning faster than random*, which this one-seed pilot does not establish.

## 15. EXP-013: read the complete results table

This table gives both metrics at every budget, so we do not choose a single favourable checkpoint. At 72 pages, random reached CER 0.175 and WER 0.553; entropy was close at 0.178 and 0.559. The starred zero-page column is the reused original-model evaluation. None of the trained checkpoints had a validation line cut off by the output limit, so that particular whole-page failure is largely removed from this comparison. Nevertheless, a small observed gap in one seed is not yet evidence of a repeatable saving in labels.

## 16. What was selected, and how much text was revealed?

The selected pages themselves help explain the results. Entropy and least confidence shared 37 of their 72 acquired pages: different formulas can lead to quite similar training sets. The methods also revealed different numbers of line transcripts at the same **page** budget. At 72 pages, random trained on 1,725 lines and k-center on 1,658. Equal page counts are the stated acquisition control, but they do not mean equal amounts of text or equal human annotation effort. I record overlap and training volume so the error curves are not interpreted more strongly than the design allows.

## 17. What is established, and what comes next?

The project now has working whole-page and known-box recognition pipelines, verified learning curves and reproducible selection records. Fine-tuning clearly improves OCR on these validation slices. What is **not** established is a repeatable active-selection advantage, independent-document generalization, box detection or human-time saving. EXP-013 used one configured seed, and the adapter initialization was not explicitly seeded before construction. My next research test is to control that initialization and repeat the 24/48/72 comparison with paired seeds and a justified document-grouped evaluation. The early entropy signal is a hypothesis to test, not the final conclusion.

## 18. Research context and close

The methods were chosen as transparent baselines: Settles explains classical uncertainty sampling, Sener and Savarese motivate core-set coverage, and Munjal and colleagues show why strong random controls and repeatability matter. Work on historical handwritten-text recognition motivates the low-label setting. These papers support the **design choices**, not our particular OCR results. The claim I can defend today is a reliable experimental path and a specific open question: under controlled repeated runs, can model-guided selection beat random acquisition early enough to reduce the labels needed for historical OCR?

---

### Evidence for questions after the talk (not part of the spoken script)

- Reader-facing experiment reports: [EXP-009](../../results/EXP-009.md), [EXP-010](../../results/EXP-010.md), [EXP-011](../../results/EXP-011.md), [EXP-012](../../results/EXP-012.md), [EXP-013](../../results/EXP-013.md).
- Technical methods, receipts and model identities: [`experiments/`](../../experiments/). The source data and model adapters are retained under the project's ignored `.local/` paths; the reports identify them.
- Method papers: [Settles, *Active Learning Literature Survey*](https://burrsettles.com/pub/settles.activelearning.pdf); [Sener and Savarese, core-set selection](https://arxiv.org/abs/1708.00489); [Munjal et al., robust active-learning comparisons](https://arxiv.org/abs/2002.09564); [Aradillas et al., few-label historical recognition](https://arxiv.org/abs/2012.02544).
