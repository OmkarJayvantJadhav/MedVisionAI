import { formatPercent } from "../lib/labels.js";
import { publishedResults } from "../lib/modelResults.js";
import { ChevronDownIcon } from "./icons.jsx";
import { REPO_URL } from "./TopBar.jsx";

const BACKBONES = {
  efficientnetb0: "EfficientNet-B0",
  efficientnetv2s: "EfficientNetV2-S",
  convnext_tiny: "ConvNeXt-Tiny",
  mobilenetv2: "MobileNetV2",
};

const MODEL_CARD_URL = `${REPO_URL}/blob/main/docs/model_card.md`;

/** Percent with one decimal when the data has it (99.6%, not a rounded 100%). */
function precisePercent(value) {
  return `${Math.round(value * 1000) / 10}%`;
}

function Panel({ id, title, summary, children }) {
  return (
    <details id={id} className="group border-b border-line last:border-b-0">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-5 py-4 marker:hidden hover:bg-sunken/50 sm:px-6 [&::-webkit-details-marker]:hidden">
        <span className="min-w-0">
          <span className="block text-[15px] font-semibold">{title}</span>
          <span className="mt-0.5 block text-sm text-ink-soft">{summary}</span>
        </span>
        <ChevronDownIcon className="h-5 w-5 shrink-0 text-ink-soft transition group-open:rotate-180" />
      </summary>
      <div className="px-5 pt-1 pb-6 text-sm leading-relaxed sm:px-6">{children}</div>
    </details>
  );
}

function Fact({ label, children }) {
  return (
    <div className="flex flex-col gap-0.5 border-t border-line py-2.5 first:border-t-0 sm:grid sm:grid-cols-[14rem_1fr] sm:gap-4">
      <dt className="text-ink-soft">{label}</dt>
      <dd className="min-w-0 break-words">{children}</dd>
    </div>
  );
}

function ModelCardLink() {
  return (
    <a
      href={MODEL_CARD_URL}
      className="font-semibold text-accent underline-offset-4 hover:underline"
    >
      model card
    </a>
  );
}

const STEPS = [
  {
    title: "Check the image",
    text: "Brightness, sharpness and sensor noise are measured, and the image's features are compared with the training images. Images that fail are not assessed.",
  },
  {
    title: "Score eight lesion types",
    text: "A neural network gives a probability for each lesion type; temperature scaling calibrates them.",
  },
  {
    title: "Answer the referral question",
    text: "Refer when the malignant and pre-malignant types together reach the referral threshold; unsure when the top answer is below the confidence threshold; otherwise no flag.",
  },
  {
    title: "Show where it looked",
    text: "A heatmap from the same forward pass shows which regions drove the score.",
  },
];

function HowItWorks() {
  return (
    <>
      <ol className="grid gap-3 sm:grid-cols-2">
        {STEPS.map((step, index) => (
          <li key={step.title} className="rounded-xl bg-sunken/60 p-4">
            <span className="font-mono text-xs text-accent">0{index + 1}</span>
            <p className="mt-1 font-semibold">{step.title}</p>
            <p className="mt-1 text-ink-soft">{step.text}</p>
          </li>
        ))}
      </ol>
      <p className="mt-4 text-ink-soft">
        Every step runs on the server&apos;s CPU with ONNX Runtime, in one request. The clinician
        makes the referral decision; the tool supports it.
      </p>
    </>
  );
}

function ModelFacts({ meta }) {
  if (!meta) {
    return <p className="text-ink-soft">Model details appear here once the server is reachable.</p>;
  }
  const gates = [];
  if (meta.thresholds?.ood_mahalanobis != null) gates.push("non-dermoscopy images");
  if (meta.thresholds?.max_noise != null) gates.push("grainy images");
  return (
    <dl>
      <Fact label="Version">
        <span className="font-mono text-[13px]">{meta.model_version}</span>
      </Fact>
      <Fact label="Architecture">
        {BACKBONES[meta.backbone] ?? meta.backbone}, {meta.image_size}×{meta.image_size} px input
      </Fact>
      <Fact label="Lesion types recognised">{meta.class_names.join(", ")}</Fact>
      <Fact label="Counted as malignant or pre-malignant">{meta.malignant_classes.join(", ")}</Fact>
      <Fact label="Calibration">
        {meta.calibrated ? "Temperature scaling, fitted on validation data" : "Not calibrated"}
      </Fact>
      <Fact label="Decision thresholds">
        {meta.thresholds.fitted ? (
          <>
            Refer at {formatPercent(meta.thresholds.malignant)} combined malignancy probability ·
            unsure below {formatPercent(meta.thresholds.low_confidence)} confidence
          </>
        ) : (
          "Provisional (not yet fitted on data)"
        )}
      </Fact>
      {gates.length > 0 && <Fact label="Rejects">{gates.join(" and ")}</Fact>}
      <Fact label="Upload limit">JPEG, PNG or WebP, up to {meta.max_upload_mb} MB</Fact>
    </dl>
  );
}

function Performance({ meta }) {
  const results = meta ? publishedResults(meta.model_version) : null;
  if (!results) {
    return (
      <p className="text-ink-soft">
        Measured results for each model version are published in the <ModelCardLink />.
      </p>
    );
  }
  const rows = [
    ["Malignant lesions flagged for referral (recall)", precisePercent(results.malignantRecall)],
    ["Flagged lesions that were malignant (precision)", precisePercent(results.malignantPrecision)],
    ["Images flagged for referral", precisePercent(results.referralRate)],
    ["Calibration error (ECE)", results.ece.toFixed(3)],
    [
      "Answered with confidence",
      `${precisePercent(results.confidentShare)}, of which ${precisePercent(results.confidentAccuracy)} correct`,
    ],
    ["Lesion images passing the input checks", precisePercent(results.lesionsAccepted)],
    ["Everyday photos rejected", precisePercent(results.photosRejected)],
    [
      "Top-1 accuracy · macro F1",
      `${precisePercent(results.accuracy)} · ${results.macroF1.toFixed(3)}`,
    ],
  ];
  return (
    <>
      <p className="text-ink-soft">
        Held-out ISIC 2019 test split ({results.testImages.toLocaleString("en")} images) for the
        served model, <span className="font-mono text-[13px]">{meta.model_version}</span>.
      </p>
      <dl className="mt-3">
        {rows.map(([label, value]) => (
          <Fact key={label} label={label}>
            <span className="tabular-nums">{value}</span>
          </Fact>
        ))}
      </dl>
      <p className="mt-3 text-ink-soft">
        The test split is 37% malignant. In primary care, where cancer is rarer, a smaller share of
        flagged lesions would be malignant. Results by sex, age and body site, and under image
        damage, are in the <ModelCardLink />.
      </p>
    </>
  );
}

const LIMITATIONS = [
  "Dermoscopy images only: clinical and smartphone photos are rejected, not assessed.",
  "Eight lesion types only. Rarer skin cancers were never seen in training, so a no-flag answer never rules them out.",
  "To catch 9 in 10 malignant lesions, the model flags a large share of lesions for referral.",
  "Performance on darker skin tones is unknown: the training data does not record skin type.",
  "Heavily compressed JPEG images can be misread without a warning.",
  "Evaluated on one dataset, without external or prospective clinical validation. Not a medical device.",
];

/** Collapsible background sections below the workspace. */
export default function InfoSections({ meta }) {
  return (
    <section id="about" aria-labelledby="about-heading" className="mx-auto max-w-5xl px-4 py-16">
      <h2 id="about-heading" className="text-xl font-bold tracking-tight">
        About MedVision AI
      </h2>
      <p className="mt-1 text-sm text-ink-soft">
        How the referral support works, what it can and cannot do, and how images are handled.
      </p>
      <div className="mt-6 overflow-hidden rounded-2xl border border-line bg-surface shadow-[var(--shadow-card)]">
        <Panel
          id="how-it-works"
          title="How MedVision AI works"
          summary="Four steps in one request: check, score, answer, explain."
        >
          <HowItWorks />
        </Panel>
        <Panel id="model" title="About the AI model" summary="Read live from the served model.">
          <ModelFacts meta={meta} />
        </Panel>
        <Panel
          id="privacy"
          title="Privacy and data handling"
          summary="Images are analysed in memory and never stored."
        >
          <ul className="list-disc space-y-1.5 pl-5 text-ink-soft">
            <li>Images are analysed in memory and discarded as soon as the result is sent.</li>
            <li>They are not written to disk, logged, or used for training.</li>
            <li>
              No accounts, tracking, analytics or third-party requests; fonts are self-hosted.
            </li>
            <li>Only your theme choice is remembered, in this browser.</li>
          </ul>
        </Panel>
        <Panel
          id="performance"
          title="Model performance"
          summary="Measured on held-out test images, never used in training."
        >
          <Performance meta={meta} />
        </Panel>
        <Panel
          id="limitations"
          title="Known limitations"
          summary="What the referral support cannot do."
        >
          <ul className="list-disc space-y-1.5 pl-5 text-ink-soft">
            {LIMITATIONS.map((limit) => (
              <li key={limit}>{limit}</li>
            ))}
          </ul>
        </Panel>
      </div>
    </section>
  );
}
