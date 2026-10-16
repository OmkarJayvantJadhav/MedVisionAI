/**
 * Keeps the frontend honest about the API shape without TypeScript: fixtures and
 * label maps are checked against the OpenAPI schema generated from the backend
 * (a backend test fails when that schema is stale).
 */
import { describe, expect, it } from "vitest";

import { RISK, STATUS } from "../lib/labels.js";
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

  it("every status and risk value has a label", () => {
    expect(Object.keys(STATUS).sort()).toEqual([...schemas.Status.enum].sort());
    expect(Object.keys(RISK).sort()).toEqual([...schemas.RiskFlag.enum].sort());
  });

  it("the endpoints the client calls exist", () => {
    expect(openapi.paths).toHaveProperty("/api/v1/explain");
    expect(openapi.paths).toHaveProperty("/api/v1/meta");
  });
});
