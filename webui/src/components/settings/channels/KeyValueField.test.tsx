import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import { KeyValueField, parseKeyValueString, serializeKeyValuePairs } from "./KeyValueField";

describe("parseKeyValueString", () => {
  it("parses multi-line key=value pairs", () => {
    expect(parseKeyValueString("a=1\nb=2")).toEqual([
      { key: "a", value: "1" },
      { key: "b", value: "2" },
    ]);
  });

  it("ignores empty lines and comments", () => {
    expect(parseKeyValueString("a=1\n\n# comment\nb=2")).toEqual([
      { key: "a", value: "1" },
      { key: "b", value: "2" },
    ]);
  });

  it("treats lines without '=' as key with empty value", () => {
    expect(parseKeyValueString("a")).toEqual([{ key: "a", value: "" }]);
  });
});

describe("serializeKeyValuePairs", () => {
  it("serializes pairs to multi-line key=value", () => {
    expect(
      serializeKeyValuePairs([
        { key: "a", value: "1" },
        { key: "b", value: "2" },
      ]),
    ).toBe("a=1\nb=2");
  });

  it("drops entries with empty keys", () => {
    expect(serializeKeyValuePairs([{ key: "", value: "1" }])).toBe("");
  });
});

describe("KeyValueField", () => {
  it("renders existing pairs", () => {
    render(<KeyValueField id="kv" value={"a=1\nb=2"} onChange={() => {}} />);
    expect(screen.getAllByRole("textbox")).toHaveLength(4);
  });

  it("does not serialize empty added pairs", () => {
    const onChange = vi.fn();
    render(<KeyValueField id="kv" value="a=1" onChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: /add/i }));
    expect(onChange).toHaveBeenCalledWith("a=1");
  });

  it("removes a pair when remove is clicked", () => {
    const onChange = vi.fn();
    render(<KeyValueField id="kv" value={"a=1\nb=2"} onChange={onChange} />);
    fireEvent.click(screen.getAllByRole("button", { name: /remove entry/i })[0]);
    expect(onChange).toHaveBeenCalledWith("b=2");
  });

  it("updates key/value inputs", () => {
    const onChange = vi.fn();
    render(<KeyValueField id="kv" value="a=1" onChange={onChange} />);
    const inputs = screen.getAllByRole("textbox");
    fireEvent.change(inputs[0], { target: { value: "x" } });
    expect(onChange).toHaveBeenCalledWith("x=1");
    fireEvent.change(inputs[1], { target: { value: "9" } });
    expect(onChange).toHaveBeenCalledWith("a=9");
  });
});
