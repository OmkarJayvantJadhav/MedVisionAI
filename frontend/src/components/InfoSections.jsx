import { formatPercent } from "../lib/labels.js";
import { publishedResults } from "../lib/modelResults.js";
import { ChevronDownIcon } from "./icons.jsx";
import { REPO_URL } from "./TopBar.jsx";

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

function ReportLink() {
  return (
    <a
      href={MODEL_CARD_URL}
      className="font-semibold text-accent underline-offset-4 hover:underline"
    >
      full test report
    </a>
  );
}

const STEPS = [
  {
    title: "Checks the picture",
    text: "Makes sure the picture is clear, well lit, not grainy and really a dermoscopy image. If it is not, the system asks for a better picture instead of guessing.",
  },
  {
    title: "Compares the lesion",
    text: "Compares the lesion with eight common types it learned from about 25,000 images labelled by skin experts.",
  },
  {
    title: "Names the type and checks the risk",
    text: "Gives the most likely of the eight types and says whether it is confident or uncertain. Separately, it raises a high-risk alert when the cancer and pre-cancer types together pass the alert line, whatever the most likely type is.",
  },
  {
    title: "Shows where it looked",
    text: "Colours the areas of the picture that mattered most for its answer.",
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
        The model supports your clinical judgement; it does not replace it.
      </p>
    </>
  );
}

function SystemFacts({ meta }) {
  if (!meta) {
    return <p className="text-ink-soft">These details appear once the system is ready.</p>;
  }
  const turnedAway = [];
  if (meta.thresholds?.ood_mahalanobis != null) turnedAway.push("Pictures that are not dermoscopy");
  if (meta.thresholds?.max_noise != null) turnedAway.push("grainy pictures");
  return (
    <dl>
      <Fact label="Lesion types it knows">{meta.class_names.join(", ")}</Fact>
      <Fact label="Counted as cancer or pre-cancer">{meta.malignant_classes.join(", ")}</Fact>
      <Fact label="When it raises a high-risk alert">
        {meta.thresholds.fitted
          ? `When the chance of cancer or pre-cancer is ${formatPercent(meta.thresholds.malignant)} or more`
          : "Not tuned yet in this version"}
      </Fact>
      <Fact label="When it says unsure">
        {meta.thresholds.fitted
          ? `When its best match is below ${formatPercent(meta.thresholds.low_confidence)}`
          : "Not tuned yet in this version"}
      </Fact>
      <Fact label="Percentages">
        {meta.calibrated
          ? "Adjusted so that a 70% match is right about 7 times in 10"
          : "Rough scores, not reliable chances"}
      </Fact>
      {turnedAway.length > 0 && <Fact label="Turns away">{turnedAway.join(" and ")}</Fact>}
      <Fact label="Pictures accepted">JPEG, PNG or WebP, up to {meta.max_upload_mb} MB</Fact>
      <Fact label="Version">
        <span className="font-mono text-[13px]">{meta.model_version}</span>
      </Fact>
    </dl>
  );
}

function Performance({ meta }) {
  const results = meta ? publishedResults(meta.model_version) : null;
  if (!results) {
    return (
      <p className="text-ink-soft">
        Test results for each version are in the <ReportLink />.
      </p>
    );
  }
  const rows = [
    ["Lesion type matched correctly", precisePercent(results.accuracy)],
    ["Melanomas named as melanoma", precisePercent(results.melanomaRecall)],
    ["Cancers and pre-cancers that raised the alert", precisePercent(results.malignantRecall)],
    ["Alerts that really were cancer or pre-cancer", precisePercent(results.malignantPrecision)],
    ["All lesions that raised the alert", precisePercent(results.referralRate)],
    [
      "Answered confidently",
      `${precisePercent(results.confidentShare)} of pictures, and ${precisePercent(results.confidentAccuracy)} of those were right`,
    ],
    [
      "How far its percentages were off",
      `About ${Math.round(results.ece * 100)} percentage points on average`,
    ],
    ["Clear dermoscopy pictures accepted", precisePercent(results.lesionsAccepted)],
    ["Everyday photos turned away", precisePercent(results.photosRejected)],
  ];
  return (
    <>
      <p className="text-ink-soft">
        Tested on {results.testImages.toLocaleString("en")} dermoscopy pictures that the system
        never saw while it was learning.
      </p>
      <dl className="mt-3">
        {rows.map(([label, value]) => (
          <Fact key={label} label={label}>
            <span className="tabular-nums">{value}</span>
          </Fact>
        ))}
      </dl>
      <p className="mt-3 text-ink-soft">
        37% of the test pictures were cancer or pre-cancer. In everyday practice, where cancer is
        rarer, fewer of the alerts would turn out to be cancer. Results for men and women, age
        groups, body areas and damaged pictures are in the <ReportLink />.
      </p>
    </>
  );
}

const LIMITATIONS = [
  "Works only with dermoscopy images. Ordinary photos, including phone photos, are turned away.",
  "Knows eight lesion types only. It never learned rarer skin cancers, so a no-flag answer never rules them out.",
  "The lesion type is right about 7 times in 10, and melanoma is named correctly less than half the time. The high-risk alert, not the lesion type, is what catches most cancers.",
  "To catch 9 in 10 cancers and pre-cancers, the alert is raised for about half of all lesions.",
  "How well it works on darker skin is unknown: the pictures it learned from do not record skin tone.",
  "Heavily compressed, low-quality JPEG pictures can be misread without a warning.",
  "Tested on one collection of pictures only, not in real clinics. It is not a medical device.",
];

/** Collapsible background sections below the workspace. */
export default function InfoSections({ meta }) {
  return (
    <section id="about" aria-labelledby="about-heading" className="mx-auto max-w-5xl px-4 py-16">
      <h2 id="about-heading" className="text-xl font-bold tracking-tight">
        About MedVision AI
      </h2>
      <p className="mt-1 text-sm text-ink-soft">
        How the check works, what it can and cannot do, and what happens to your pictures.
      </p>
      <div className="mt-6 overflow-hidden rounded-2xl border border-line bg-surface shadow-[var(--shadow-card)]">
        <Panel
          id="how-it-works"
          title="How MedVision AI works"
          summary="Four steps: check, compare, answer, show."
        >
          <HowItWorks />
        </Panel>
        <Panel
          id="model"
          title="What the system knows"
          summary="Lesion types, when it raises an alert, and accepted pictures."
        >
          <SystemFacts meta={meta} />
        </Panel>
        <Panel id="privacy" title="Privacy" summary="Pictures are never saved.">
          <ul className="list-disc space-y-1.5 pl-5 text-ink-soft">
            <li>Pictures are checked and deleted as soon as the result is shown.</li>
            <li>They are never saved, recorded or used to teach the system.</li>
            <li>No accounts and no tracking, and nothing is sent to other companies.</li>
            <li>Only your light or dark choice is remembered, in this browser.</li>
          </ul>
        </Panel>
        <Panel
          id="performance"
          title="How well it works"
          summary="Tested on pictures the system never saw while learning."
        >
          <Performance meta={meta} />
        </Panel>
        <Panel id="limitations" title="Limitations" summary="What the system cannot do.">
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
