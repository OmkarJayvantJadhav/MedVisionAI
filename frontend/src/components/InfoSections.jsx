import { formatPercent } from "../lib/labels.js";
import { LockIcon } from "./icons.jsx";
import { REPO_URL } from "./TopBar.jsx";

const BACKBONES = {
  efficientnetb0: "EfficientNet-B0",
  efficientnetv2s: "EfficientNetV2-S",
  convnext_tiny: "ConvNeXt-Tiny",
  mobilenetv2: "MobileNetV2",
};

function Eyebrow({ children }) {
  return (
    <p className="text-xs font-semibold tracking-[0.14em] text-accent uppercase">{children}</p>
  );
}

const STEPS = [
  {
    title: "Check the photo",
    text: "Dark, blurry, grainy or non-dermoscopy images are turned away instead of guessed at.",
  },
  {
    title: "Classify",
    text: "A neural network scores eight conditions; temperature scaling keeps the percentages honest.",
  },
  {
    title: "Apply safety rules",
    text: "Unsure answers are marked as such, and a malignancy flag fires when cancer-related classes add up.",
  },
  {
    title: "Explain",
    text: "A heatmap from the same forward pass shows which regions drove the score.",
  },
];

export function HowItWorks() {
  return (
    <section id="how-it-works" className="mx-auto max-w-6xl px-4 py-20 sm:py-24">
      <div className="max-w-2xl">
        <Eyebrow>How it works</Eyebrow>
        <h2 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">
          Four steps, a few seconds.
        </h2>
        <p className="mt-3 text-[15px] leading-relaxed text-ink-soft">
          Everything runs on the server&apos;s CPU with ONNX Runtime. Nothing is sent to third
          parties.
        </p>
      </div>
      <ol className="mt-12 grid gap-x-8 gap-y-10 sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map((step, index) => (
          <li key={step.title} className="border-t border-line pt-5">
            <span className="font-mono text-xs text-accent">0{index + 1}</span>
            <h3 className="mt-2 text-base font-semibold">{step.title}</h3>
            <p className="mt-1.5 text-sm leading-relaxed text-ink-soft">{step.text}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

function Fact({ label, children }) {
  return (
    <div className="flex flex-col gap-1 border-t border-line py-4 sm:grid sm:grid-cols-[13rem_1fr] sm:gap-6">
      <dt className="text-sm text-ink-soft">{label}</dt>
      <dd className="text-sm">{children}</dd>
    </div>
  );
}

/** Facts read from the served model's metadata, so they always match it. */
export function ModelFacts({ meta }) {
  const gates = [];
  if (meta?.thresholds?.ood_mahalanobis != null) gates.push("non-dermoscopy images");
  if (meta?.thresholds?.max_noise != null) gates.push("grainy images");
  return (
    <section id="model" className="border-y border-line bg-surface/70">
      <div className="mx-auto grid max-w-6xl gap-10 px-4 py-20 sm:py-24 lg:grid-cols-[1fr_1.4fr]">
        <div className="max-w-md">
          <Eyebrow>The model</Eyebrow>
          <h2 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">
            Open about what it is.
          </h2>
          <p className="mt-3 text-[15px] leading-relaxed text-ink-soft">
            Results, calibration plots, robustness tests and known limitations are published in the
            model card, and every number can be reproduced from the repository.
          </p>
          <a
            href={`${REPO_URL}/blob/main/docs/model_card.md`}
            className="mt-5 inline-flex items-center gap-1.5 text-sm font-semibold text-accent underline-offset-4 hover:underline"
          >
            Read the model card <span aria-hidden="true">→</span>
          </a>
        </div>
        {meta ? (
          <dl>
            <Fact label="Version">
              <span className="font-mono text-[13px]">{meta.model_version}</span>
            </Fact>
            <Fact label="Architecture">
              {BACKBONES[meta.backbone] ?? meta.backbone}, {meta.image_size}×{meta.image_size} px
              input
            </Fact>
            <Fact label="Conditions recognised">{meta.class_names.join(", ")}</Fact>
            <Fact label="Counted as cancer-related">{meta.malignant_classes.join(", ")}</Fact>
            <Fact label="Calibration">
              {meta.calibrated
                ? "Temperature scaling, fitted on validation data"
                : "Not calibrated"}
            </Fact>
            <Fact label="Decision thresholds">
              {meta.thresholds.fitted ? (
                <>
                  Unsure below {formatPercent(meta.thresholds.low_confidence)} confidence ·
                  malignancy flag at {formatPercent(meta.thresholds.malignant)} combined
                </>
              ) : (
                "Provisional (not yet fitted on data)"
              )}
            </Fact>
            {gates.length > 0 && <Fact label="Rejects">{gates.join(" and ")}</Fact>}
          </dl>
        ) : (
          <p className="self-center text-sm text-ink-faint">
            Model details appear here once the server is reachable.
          </p>
        )}
      </div>
    </section>
  );
}

const LIMITS = [
  "It only learned from dermoscopy images (magnified, evenly lit), so ordinary phone photos are turned away.",
  "Accuracy on darker skin tones is unknown: the training data does not record skin type.",
  "It only knows the eight conditions above, not rashes, infections or anything else.",
];

export function Privacy() {
  return (
    <section id="privacy" className="mx-auto max-w-6xl px-4 py-20 sm:py-24">
      <div className="grid gap-12 lg:grid-cols-2">
        <div>
          <Eyebrow>Privacy</Eyebrow>
          <h2 className="mt-3 text-3xl font-bold tracking-tight">Your photo is never stored.</h2>
          <p className="mt-4 flex gap-3 text-[15px] leading-relaxed text-ink-soft">
            <LockIcon className="mt-1 h-5 w-5 shrink-0 text-accent" />
            Images are analysed in memory and discarded as soon as the result is sent. They are not
            written to disk, logged, or used for training. No accounts, no tracking.
          </p>
        </div>
        <div>
          <Eyebrow>Limitations</Eyebrow>
          <h2 className="mt-3 text-3xl font-bold tracking-tight">What it cannot do.</h2>
          <ul className="mt-4 space-y-3">
            {LIMITS.map((limit) => (
              <li key={limit} className="flex gap-3 text-[15px] leading-relaxed text-ink-soft">
                <span
                  aria-hidden="true"
                  className="mt-2.5 h-1.5 w-1.5 shrink-0 rounded-full bg-ink-faint"
                />
                {limit}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
