import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => cleanup());

// jsdom has no object URLs; previews only need a stable string.
globalThis.URL.createObjectURL ??= () => "blob:preview";
globalThis.URL.revokeObjectURL ??= () => {};
