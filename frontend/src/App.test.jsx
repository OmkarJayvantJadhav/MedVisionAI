import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App.jsx";
import { explainResponse, imageFile, jsonResponse, metaResponse } from "./test/fixtures.js";

function mockApi({ explain = jsonResponse(explainResponse) } = {}) {
  const fetchMock = vi.fn((url) =>
    Promise.resolve(url.endsWith("/meta") ? jsonResponse(metaResponse) : explain),
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

async function chooseFile(user, file = imageFile()) {
  await user.upload(screen.getByLabelText("Choose an image file"), file);
}

beforeEach(() => vi.clearAllMocks());
afterEach(() => vi.unstubAllGlobals());

describe("App", () => {
  it("always shows the disclaimer and loads model metadata", async () => {
    mockApi();
    render(<App />);
    expect(screen.getByRole("note", { name: "Medical disclaimer" })).toBeInTheDocument();
    expect(await screen.findByText(/Model skin-effb0-v2.0.0/)).toBeInTheDocument();
  });

  it("analyses an uploaded image and shows the result", async () => {
    const user = userEvent.setup();
    const fetchMock = mockApi();
    render(<App />);

    const analyse = screen.getByRole("button", { name: "Analyse image" });
    expect(analyse).toBeDisabled();
    await chooseFile(user);
    expect(screen.getByAltText("Selected skin lesion")).toBeInTheDocument();

    await user.click(analyse);
    expect(await screen.findByText("Assessment complete")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/explain",
      expect.objectContaining({ method: "POST" }),
    );

    await user.click(screen.getByRole("button", { name: "Choose another" }));
    expect(screen.queryByText("Assessment complete")).not.toBeInTheDocument();
    expect(screen.getByText(/result and heatmap will appear here/)).toBeInTheDocument();
  });

  it("rejects unsupported files before uploading", async () => {
    const user = userEvent.setup({ applyAccept: false });
    const fetchMock = mockApi();
    render(<App />);

    await chooseFile(user, imageFile("anim.gif", "image/gif"));
    expect(screen.getByRole("alert")).toHaveTextContent(/Only JPEG, PNG and WebP/);
    expect(fetchMock).not.toHaveBeenCalledWith("/api/v1/explain", expect.anything());
  });

  it("explains server errors and can dismiss them", async () => {
    const user = userEvent.setup();
    mockApi({
      explain: jsonResponse({ error: { code: "model_unavailable", message: "x" } }, 503),
    });
    render(<App />);

    await chooseFile(user);
    await user.click(screen.getByRole("button", { name: "Analyse image" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/model is still loading/);
    expect(screen.getByRole("button", { name: "Analyse image" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "Dismiss" }));
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
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
    await user.click(screen.getByRole("button", { name: "Analyse image" }));
    expect(await screen.findByText("Assessment complete")).toBeInTheDocument();
  });
});
