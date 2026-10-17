import { formatPercent } from "../lib/labels.js";
import { LockIcon } from "./icons.jsx";

const BACKBONES = {
  efficientnetb0: "EfficientNet-B0",
  efficientnetv2s: "EfficientNetV2-S",
  convnext_tiny: "ConvNeXt-Tiny",
  mobilenetv2: "MobileNetV2",
};
import { REPO_URL } from "./TopBar.jsx";

function SectionHeading({ eyebrow, title, children }) {
  return (
    <div className="max-w-2xl">
      <p className="text-xs font-semibold tracking-[0.1em] text-accent uppercase">{eyebrow}</p>
      <h2 className="mt-2 text-2xl font-bold tracking-tight sm:text-3xl">{title}</h2>
      {children && <p className="mt-3 text-[15px] leading-relaxed text-ink-soft">{children}</p>}
    </div>
  );
}

const STEPS = [
  {
    title: "Quality check",
    text: "The photo is decoded in memory and checked for darkness, glare and blur. Unusable photos are rejected instead of guessed at.",
  },
  {
    title: "Classification",
    text: "A convolutional neural network scores every condition it was trained on, and temperature scaling calibrates those scores into honest probabilities.",
  },
  {
    title: "Safety rules",
    text: "Answers below a confidence threshold are marked as unsure, and a malignancy flag is raised whenever cancer-related classes add up past a second threshold.",
  },
  {
    title: "Explanation",
    text: "A class activation map shows which regions drove the score, computed from the same forward pass in milliseconds.",
  },
];

export function HowItWorks() {
  return (
    <section id="how-it-works" className="mx-auto max-w-6xl px-4 py-16 sm:py-20">
      <SectionHeading eyebrow="How it works" title="From photo to explained result in seconds">
        Every step runs on the server&apos;s CPU with ONNX Runtime. Nothing is sent to third
        parties.
      </SectionHeading>
      <ol className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map((step, index) => (
          <li key={step.title} className="rounded-2xl border border-line bg-surface p-5">
            <span className="font-mono text-xs font-medium text-accent">Step {index + 1}</span>
            <h3 className="mt-2 text-base font-semibold">{step.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-ink-soft">{step.text}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

function Fact({ label, children }) {
  return (
    <div className="border-t border-line py-3.5 sm:grid sm:grid-cols-[14rem_1fr] sm:gap-6">
      <dt className="text-sm font-medium text-ink-soft">{label}</dt>
      <dd className="mt-1 text-sm sm:mt-0">{children}</dd>
    </div>
  );
}

/** Facts read from the served model's metadata, so they always match it. */
export function ModelFacts({ meta }) {
  return (
    <section id="model" className="border-y border-line bg-surface">
      <div className="mx-auto grid max-w-6xl gap-10 px-4 py-16 sm:py-20 lg:grid-cols-[1fr_1.3fr]">
        <SectionHeading eyebrow="The model" title="Transparent about what it is">
          Test-set results, calibration plots and the training recipe are published with each model
          release, and every number can be reproduced from the repository.
          <a
            href={`${REPO_URL}/releases`}
            className="mt-4 block font-semibold text-accent underline-offset-4 hover:underline"
          >
            Read the model release notes
          </a>
        </SectionHeading>
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
            <Fact label="Counted as malignant or pre-malignant">
              {meta.malignant_classes.join(", ")}
            </Fact>
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
  "Trained on dermoscopy (magnified, evenly lit) images. Ordinary phone photos are harder for it.",
  "The training data under-represents darker skin tones, so accuracy there is not established.",
  "It only knows the conditions listed above and cannot recognise anything else, such as rashes or infections.",
];

export function Privacy() {
  return (
    <section id="privacy" className="mx-auto max-w-6xl px-4 py-16 sm:py-20">
      <div className="grid gap-10 lg:grid-cols-2">
        <div>
          <SectionHeading eyebrow="Privacy" title="Your photo is never stored" />
          <div className="mt-6 flex gap-4 rounded-2xl border border-line bg-surface p-5">
            <LockIcon className="mt-0.5 h-6 w-6 shrink-0 text-accent" />
            <p className="text-sm leading-relaxed text-ink-soft">
              Images are analysed in memory and discarded as soon as the result is sent. They are
              not written to disk, logged, or used for training. There are no accounts and no
              tracking.
            </p>
          </div>
        </div>
        <div>
          <SectionHeading eyebrow="Limitations" title="What this tool cannot do" />
          <ul className="mt-6 space-y-3">
            {LIMITS.map((limit) => (
              <li
                key={limit}
                className="rounded-2xl border border-line bg-surface p-4 text-sm leading-relaxed text-ink-soft"
              >
                {limit}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
