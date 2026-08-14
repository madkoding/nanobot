import { useMemo } from "react";
import { Plus, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

export type KeyValuePair = { key: string; value: string };

export function parseKeyValueString(text: string): KeyValuePair[] {
  const pairs: KeyValuePair[] = [];
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const separatorIndex = trimmed.indexOf("=");
    if (separatorIndex === -1) {
      // Treat the whole line as a key with empty value; the UI lets the user
      // fill the value input next to it.
      pairs.push({ key: trimmed, value: "" });
      continue;
    }
    pairs.push({
      key: trimmed.slice(0, separatorIndex).trim(),
      value: trimmed.slice(separatorIndex + 1).trim(),
    });
  }
  return pairs;
}

export function serializeKeyValuePairs(pairs: KeyValuePair[]): string {
  return pairs
    .map(({ key, value }) => {
      const k = key.trim();
      if (!k) return "";
      return `${k}=${value.trim()}`;
    })
    .filter(Boolean)
    .join("\n");
}

export function KeyValueField({
  id,
  value,
  placeholder,
  keyPlaceholder,
  valuePlaceholder,
  keyAriaLabel,
  valueAriaLabel,
  addLabel,
  removeLabel,
  compact = false,
  onChange,
}: {
  id: string;
  value: string;
  placeholder?: string;
  keyPlaceholder?: string;
  valuePlaceholder?: string;
  keyAriaLabel?: string;
  valueAriaLabel?: string;
  addLabel?: string;
  removeLabel?: string;
  compact?: boolean;
  onChange: (next: string) => void;
}) {
  const pairs = useMemo(() => parseKeyValueString(value), [value]);

  const update = (nextPairs: KeyValuePair[]) => {
    onChange(serializeKeyValuePairs(nextPairs));
  };

  const setPair = (index: number, patch: Partial<KeyValuePair>) => {
    const next = pairs.map((pair, i) => (i === index ? { ...pair, ...patch } : pair));
    update(next);
  };

  const removePair = (index: number) => {
    const next = pairs.filter((_, i) => i !== index);
    update(next);
  };

  const addPair = () => {
    update([...pairs, { key: "", value: "" }]);
  };

  return (
    <div
      id={id}
      className={cn(
        "rounded-[10px] border border-border/60 bg-muted/35 p-2",
        compact ? "space-y-2" : "space-y-2.5",
      )}
    >
      {pairs.length === 0 ? (
        <div className="text-[12px] text-muted-foreground">
          {placeholder ?? "No entries. Add one to get started."}
        </div>
      ) : (
        pairs.map((pair, index) => (
          <div key={index} className="flex items-start gap-2">
            <Input
              aria-label={keyAriaLabel ?? "Key"}
              type="text"
              placeholder={keyPlaceholder ?? "key"}
              value={pair.key}
              onChange={(event) => setPair(index, { key: event.target.value })}
              className="h-8 flex-1 rounded-[8px] border-border/60 bg-background/70 px-2.5 text-[12px] placeholder:text-muted-foreground/50"
            />
            <span className="mt-1.5 text-[13px] font-medium text-muted-foreground">=</span>
            <Input
              aria-label={valueAriaLabel ?? "Value"}
              type="text"
              placeholder={valuePlaceholder ?? "value"}
              value={pair.value}
              onChange={(event) => setPair(index, { value: event.target.value })}
              className="h-8 flex-1 rounded-[8px] border-border/60 bg-background/70 px-2.5 text-[12px] placeholder:text-muted-foreground/50"
            />
            <Button
              type="button"
              variant="ghost"
              size="sm"
              aria-label={removeLabel ?? "Remove entry"}
              onClick={() => removePair(index)}
              className="h-8 w-8 shrink-0 rounded-full p-0 text-muted-foreground hover:text-destructive"
            >
              <X className="h-3.5 w-3.5" aria-hidden />
            </Button>
          </div>
        ))
      )}
      <div className="flex justify-end pt-0.5">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={addPair}
          className="h-7 gap-1 rounded-full px-2 text-[12px] font-medium text-muted-foreground hover:text-foreground"
        >
          <Plus className="h-3.5 w-3.5" aria-hidden />
          {addLabel ?? "Add"}
        </Button>
      </div>
    </div>
  );
}
