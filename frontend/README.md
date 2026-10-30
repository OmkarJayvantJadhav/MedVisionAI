# medvision-frontend

React (JavaScript, JSX) + Vite + Tailwind single-page app for MedVision AI.
One workspace card handles the whole flow: drop, browse or paste a dermoscopy
image (or pick a sample), watch it being analysed, then read the referral
decision, the clinical summary, the malignancy gauge with the referral
threshold, the closest classes and the heatmap layered on the image. The
header shows model readiness and a light/dark switch, the disclaimer is always
visible, and phones show the decision before the image.

## Develop

```bash
npm ci               # Node 24 (see .nvmrc; Node 22.22.2+ also works)
npm run dev          # http://localhost:5173, proxies /api to http://localhost:8000
```

Start the API in another terminal with `make run` from the repository root.

## Check

```bash
npm run lint         # ESLint + Prettier
npm test -- --run    # Vitest + Testing Library (add --coverage for a report)
npm run build        # production build in dist/
```

The production build is served by the FastAPI app when
`MEDVISION_STATIC_DIR` points at `dist/`, so the whole product runs as one
service.

## API contract

`src/api/openapi.json` is generated from the backend (`make api-schema`).
`src/api/contract.test.js` checks the test fixtures and the status and risk
labels against it, and a backend test fails when the file is out of date, so
API changes cannot silently break the UI.

## Structure

| Path                | Purpose                                                                                               |
| ------------------- | ----------------------------------------------------------------------------------------------------- |
| `src/App.jsx`       | Page layout, readiness from `/meta`, and the workspace card's states: empty, analysing, result, error |
| `src/api/client.js` | `fetch` wrapper; every failure becomes an `ApiError` with the backend's error code                    |
| `src/components/`   | Header, upload panel, analysis progress, heatmap viewer, result card, information sections            |
| `src/lib/`          | Referral-answer wording; published results by model version; samples; theme; file checks              |
