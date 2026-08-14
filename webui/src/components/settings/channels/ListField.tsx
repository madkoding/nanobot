import { useMemo, useState } from "react";
import { Plus, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

export function parseListValue(value: string): string[] {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export function serializeListValue(items: string[]): string {
  return items.map((item) => item.trim()).filter(Boolean).join(", ");
}

export function ListField({
  id,
  value,
  placeholder,
  addLabel,
  removeLabel,
  compact = false,
  onChange,
}: {
  id: string;
  value: string;
  placeholder?: string;
  addLabel?: string;
  removeLabel?: string;
  compact?: boolean;
  onChange: (next: string) => void;
}) {
  const items = useMemo(() => parseListValue(value), [value]);
  const [draft, setDraft] = useState("");

  const commitDraft = () => {
    const next = draft.trim();
    if (!next) return;
    if (items.includes(next)) {
      setDraft("");
      return;
    }
    onChange(serializeListValue([...items, next]));
    setDraft("");
  };

  const removeItem = (index: number) => {
    const next = items.filter((_, i) => i !== index);
    onChange(serializeListValue(next));
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      event.preventDefault();
      commitDraft();
    }
    if (event.key === "Backspace" && draft === "" && items.length > 0) {
      removeItem(items.length - 1);
    }
  };

  return (
    <div
      id={id}
      className={cn(
        "rounded-[10px] border border-border/60 bg-muted/35 p-2",
        compact ? "space-y-2" : "space-y-2.5",
      )}
    >
      {items.length > 0 ? (
        <div className="flex flex-wrap gap-1.5">
          {items.map((item, index) => (
            <span
              key={`${item}-${index}`}
              className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2.5 py-1 text-[12px] font-medium text-primary ring-1 ring-inset ring-primary/20"
            >
              {item}
              <button
                type="button"
                aria-label={`${removeLabel ?? "Remove"} ${item}`}
                onClick={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  removeItem(index);
                }}
                className="ml-0.5 rounded-full p-0.5 text-primary/70 hover:text-destructive"
              >
                <X className="h-3 w-3" aria-hidden />
              </button>
            </span>
          ))}
        </div>
      ) : (
        <div className="text-[12px] text-muted-foreground/70">
          {placeholder ?? "No entries. Add one to get started."}
        </div>
      )}
      <div className="flex items-center gap-2">
        <Input
          type="text"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder ?? "Add an item and press Enter"}
          className="h-8 flex-1 rounded-[8px] border-border/60 bg-background/70 px-2.5 text-[12px] placeholder:text-muted-foreground/50"
        />
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={commitDraft}
          disabled={!draft.trim()}
          className="h-8 gap-1 rounded-full px-2 text-[12px] font-medium text-muted-foreground hover:text-foreground disabled:opacity-50"
        >
          <Plus className="h-3.5 w-3.5" aria-hidden />
          {addLabel ?? "Add"}
        </Button>
      </div>
    </div>
  );
}
