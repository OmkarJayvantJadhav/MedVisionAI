import { REPO_URL } from "./TopBar.jsx";

export default function Footer({ modelVersion }) {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-8 text-sm text-ink-soft sm:flex-row sm:items-center sm:justify-between">
        <p>
          MedVision AI · Referral decision support, not a medical device
          {modelVersion && (
            <>
              {" "}
              · Model <span className="font-mono text-[13px]">{modelVersion}</span>
            </>
          )}
        </p>
        <p className="flex flex-wrap gap-x-4 gap-y-1">
          <a
            href={`${REPO_URL}/blob/main/docs/model_card.md`}
            className="underline-offset-4 hover:text-ink hover:underline"
          >
            Model card
          </a>
          <a
            href={`${REPO_URL}/blob/main/docs/DATASETS.md`}
            className="underline-offset-4 hover:text-ink hover:underline"
          >
            Training data and licences
          </a>
          <a href={REPO_URL} className="underline-offset-4 hover:text-ink hover:underline">
            Source code
          </a>
        </p>
      </div>
    </footer>
  );
}
