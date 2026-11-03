/**
 * Keeps the frontend honest about the API shape without TypeScript: fixtures and
 * wording are checked against the OpenAPI schema generated from the backend
 * (a backend test fails when that schema is stale).
 */
import { describe, expect, it } from "vitest";

import { REASONS, verdict } from "../lib/labels.js";
import { explainResponse, metaResponse } from "../test/fixtures.js";
import openapi from "./openapi.json";

const schemas = openapi.components.schemas;

function checkAgainst(schemaName, value) {
  const schema = schemas[schemaName];
  for (const field of schema.required ?? []) {
    expect(value, `${schemaName}.${field}`).toHaveProperty(field);
  }
  for (const key of Object.keys(value)) {
    expect(Object.keys(schema.properties), `unexpected ${schemaName}.${key}`).toContain(key);
  }
}

describe("API contract", () => {
  it("explain fixture matches ExplanationResponse", () => {
    checkAgainst("ExplanationResponse", explainResponse);
    checkAgainst("ClassProbability", explainResponse.top_k[0]);
  });

  it("meta fixture matches MetaResponse", () => {
    checkAgainst("MetaResponse", metaResponse);
    checkAgainst("ThresholdsOut", metaResponse.thresholds);
  });

  it("every status and risk value gets a plain answer, and every reason has wording", () => {
    const decisions = ["Refer", "Unsure", "No flag", "Not checked"];
    for (const status of schemas.Status.enum) {
      for (const risk_flag of schemas.RiskFlag.enum) {
        expect(decisions).toContain(verdict({ status, risk_flag, reasons: [] }).decision);
      }
    }
    expect(Object.keys(REASONS).sort()).toEqual([...schemas.Reason.enum].sort());
  });

  it("the endpoints the client calls exist", () => {
    expect(openapi.paths).toHaveProperty("/api/v1/explain");
    expect(openapi.paths).toHaveProperty("/api/v1/meta");
  });
});
