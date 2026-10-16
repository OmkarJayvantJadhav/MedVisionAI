# medvision-frontend

React (JavaScript, JSX) + Vite + Tailwind single-page app for MedVision AI.
Upload or photograph a skin lesion, then see the status and risk badges, the
top matches, a plain-language summary and a Grad-CAM heatmap with adjustable
opacity. The disclaimer is always visible.

## Develop

```bash
npm ci
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

| Path                | Purpose                                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------- |
| `src/App.jsx`       | Page layout and upload → analyse → result flow                                                                  |
| `src/api/client.js` | `fetch` wrapper; every failure becomes an `ApiError` with the backend's error code                              |
| `src/components/`   | Upload panel (drag and drop, file picker, camera), result card, heatmap viewer, badges, disclaimer, error alert |
| `src/lib/`          | Labels for statuses, risk flags and error codes; client-side file checks                                        |
