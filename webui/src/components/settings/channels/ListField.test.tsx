import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import { ListField, parseListValue, serializeListValue } from "./ListField";

describe("parseListValue", () => {
  it("parses comma-separated values", () => {
    expect(parseListValue("a, b, c")).toEqual(["a", "b", "c"]);
  });

  it("ignores empty entries", () => {
    expect(parseListValue("a,, ,b")).toEqual(["a", "b"]);
  });
});

describe("serializeListValue", () => {
  it("serializes items with comma+space", () => {
    expect(serializeListValue(["a", "b"])).toBe("a, b");
  });

  it("drops empty items", () => {
    expect(serializeListValue(["a", "", "b"])).toBe("a, b");
  });
});

describe("ListField", () => {
  it("renders existing items as chips", () => {
    render(<ListField id="list" value="a, b" onChange={() => {}} />);
    expect(screen.getByText("a")).toBeDefined();
    expect(screen.getByText("b")).toBeDefined();
  });

  it("adds an item when clicking Add", () => {
    const onChange = vi.fn();
    render(<ListField id="list" value="" onChange={onChange} />);
    const input = screen.getByRole("textbox");
    fireEvent.change(input, { target: { value: "new" } });
    fireEvent.click(screen.getByRole("button", { name: /add/i }));
    expect(onChange).toHaveBeenCalledWith("new");
  });

  it("adds an item on Enter", () => {
    const onChange = vi.fn();
    render(<ListField id="list" value="" onChange={onChange} />);
    const input = screen.getByRole("textbox");
    fireEvent.change(input, { target: { value: "item" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onChange).toHaveBeenCalledWith("item");
  });

  it("removes an item when clicking its remove button", () => {
    const onChange = vi.fn();
    render(<ListField id="list" value="a, b" onChange={onChange} />);
    fireEvent.click(screen.getAllByRole("button", { name: /remove/i })[0]);
    expect(onChange).toHaveBeenCalledWith("b");
  });

  it("removes the last item on backspace when input is empty", () => {
    const onChange = vi.fn();
    render(<ListField id="list" value="a, b" onChange={onChange} />);
    const input = screen.getByRole("textbox");
    fireEvent.keyDown(input, { key: "Backspace" });
    expect(onChange).toHaveBeenCalledWith("a");
  });

  it("does not duplicate existing items", () => {
    const onChange = vi.fn();
    render(<ListField id="list" value="a" onChange={onChange} />);
    const input = screen.getByRole("textbox");
    fireEvent.change(input, { target: { value: "a" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onChange).not.toHaveBeenCalled();
  });
});
