import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App.jsx";
import { explainResponse, imageFile, jsonResponse, metaResponse } from "./test/fixtures.js";

const VERDICT = "High-risk alert: possible skin cancer";

function mockApi({ explain = () => jsonResponse(explainResponse) } = {}) {
  const fetchMock = vi.fn((url) => {
    if (url.endsWith("/meta")) return Promise.resolve(jsonResponse(metaResponse));
    if (url.endsWith(".jpg"))
      return Promise.resolve(
        new Response(new Uint8Array(64), { headers: { "Content-Type": "image/jpeg" } }),
      );
    return Promise.resolve(explain());
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const explainCalls = (fetchMock) =>
  fetchMock.mock.calls.filter(([url]) => url === "/api/v1/explain");

async function chooseFile(user, file = imageFile()) {
  await user.upload(screen.getByLabelText("Choose an image file"), file);
}

beforeEach(() => vi.clearAllMocks());
afterEach(() => vi.unstubAllGlobals());

describe("App", () => {
  it("always shows the disclaimer and the system facts from metadata", async () => {
    mockApi();
    render(<App />);
    expect(
      screen.getByRole("heading", { level: 1, name: "Dermoscopy Lesion Classifier" }),
    ).toBeVisible();
    expect(screen.getByRole("note", { name: "Medical disclaimer" })).toBeInTheDocument();
    const facts = await screen.findByText("Lesion types it knows");
    expect(facts.closest("dl")).toHaveTextContent("Melanoma, Melanocytic Nevus, Benign Keratosis");
    expect(screen.getAllByText("Not tuned yet in this version")).toHaveLength(2);
    expect(screen.getAllByText("skin-effb0-v2.0.0").length).toBeGreaterThan(0);
  });

  it("analyses a chosen photo straight away and can start over", async () => {
    const user = userEvent.setup();
    const fetchMock = mockApi();
    render(<App />);

    expect(screen.getByText("Drag a dermoscopy image here")).toBeInTheDocument();
    await chooseFile(user);
    expect(screen.getByAltText("Selected skin lesion")).toBeInTheDocument();
    expect(await screen.findByText(VERDICT)).toBeInTheDocument();
    // The heatmap is layered over the user's own photo in the same viewer.
    expect(screen.getByAltText(/Coloured highlight/)).toBeInTheDocument();
    expect(screen.getAllByAltText("Selected skin lesion")).toHaveLength(1);
    expect(explainCalls(fetchMock)).toHaveLength(1);
    expect(explainCalls(fetchMock)[0][1]).toEqual(expect.objectContaining({ method: "POST" }));

    await user.click(screen.getByRole("button", { name: "Check another" }));
    expect(screen.queryByText(VERDICT)).not.toBeInTheDocument();
    expect(screen.getByText("Drag a dermoscopy image here")).toBeInTheDocument();
  });

  it("analyses a sample image", async () => {
    const user = userEvent.setup();
    const fetchMock = mockApi();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /Basal cell carcinoma/ }));
    expect(await screen.findByText(VERDICT)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/samples/basal-cell-carcinoma.jpg", expect.anything());
    const upload = explainCalls(fetchMock)[0][1].body.get("file");
    expect(upload.name).toBe("basal-cell-carcinoma.jpg");
  });

  it("analyses an image pasted from the clipboard", async () => {
    const fetchMock = mockApi();
    render(<App />);
    const pasted = imageFile("pasted.png");
    fireEvent(window, Object.assign(new Event("paste"), { clipboardData: { files: [pasted] } }));
    expect(await screen.findByText(VERDICT)).toBeInTheDocument();
    expect(explainCalls(fetchMock)[0][1].body.get("file")).toBe(pasted);
  });

  it("rejects unsupported files before uploading", async () => {
    const user = userEvent.setup({ applyAccept: false });
    const fetchMock = mockApi();
    render(<App />);

    await chooseFile(user, imageFile("anim.gif", "image/gif"));
    expect(screen.getByRole("alert")).toHaveTextContent(/Only JPEG, PNG and WebP/);
    expect(explainCalls(fetchMock)).toHaveLength(0);
  });

  it("explains server errors, retries and can dismiss them", async () => {
    const user = userEvent.setup();
    let calls = 0;
    const fetchMock = mockApi({
      explain: () =>
        calls++ === 0
          ? jsonResponse({ error: { code: "model_unavailable", message: "x" } }, 503)
          : jsonResponse(explainResponse),
    });
    render(<App />);

    await chooseFile(user);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/still starting up/);

    await user.click(within(alert).getByRole("button", { name: "Try again" }));
    expect(await screen.findByText(VERDICT)).toBeInTheDocument();
    expect(explainCalls(fetchMock)).toHaveLength(2);
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  });

  it("can dismiss an error", async () => {
    const user = userEvent.setup({ applyAccept: false });
    mockApi();
    render(<App />);
    await chooseFile(user, imageFile("anim.gif", "image/gif"));
    await user.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("shows readiness in the header without the version number", async () => {
    mockApi();
    render(<App />);
    expect(screen.getByRole("status")).toHaveTextContent("Starting up");
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Ready"));
    expect(screen.getByRole("status")).not.toHaveTextContent("skin-effb0-v2.0.0");
  });

  it("passes the alert line from metadata to the result", async () => {
    const user = userEvent.setup();
    const fitted = { ...metaResponse, thresholds: { ...metaResponse.thresholds, fitted: true } };
    vi.stubGlobal(
      "fetch",
      vi.fn((url) =>
        Promise.resolve(
          jsonResponse(
            url.endsWith("/meta") ? fitted : { ...explainResponse, thresholds_fitted: true },
          ),
        ),
      ),
    );
    render(<App />);
    await screen.findByText("Ready");
    await chooseFile(user);
    expect(await screen.findByText(/Alert line at 30%/)).toBeInTheDocument();
  });

  it("still works when metadata cannot be loaded", async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      "fetch",
      vi.fn((url) =>
        url.endsWith("/meta")
          ? Promise.reject(new TypeError("offline"))
          : Promise.resolve(jsonResponse(explainResponse)),
      ),
    );
    render(<App />);
    await chooseFile(user);
    expect(await screen.findByText(VERDICT)).toBeInTheDocument();
    expect(screen.getByText(/These details appear once the system is ready/)).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Not available right now");
  });
});
