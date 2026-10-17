# medvision-frontend

React (JavaScript, JSX) + Vite + Tailwind single-page app for MedVision AI.
One card handles the whole flow: drop, browse, paste or photograph a lesion
(or pick a sample), see it scanned, then read the verdict, the top matches, the
cancer-related gauge and a heatmap layered on the photo. The disclaimer is
always visible, and phones show the verdict before the photo.

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

| Path                | Purpose                                                                                                   |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| `src/App.jsx`       | Page layout and the studio card's states: empty, analysing, result, error                                 |
| `src/api/client.js` | `fetch` wrapper; every failure becomes an `ApiError` with the backend's error code                        |
| `src/components/`   | Drop zone and samples, photo and heatmap viewer, result, progress, info sections, disclaimer, error alert |
| `src/lib/`          | Verdict priority and wording for statuses, reasons and errors; samples; client-side file checks           |
