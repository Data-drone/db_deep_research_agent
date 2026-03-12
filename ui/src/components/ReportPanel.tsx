import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize from "rehype-sanitize";

interface Props {
  content: string;
}

interface Section {
  title: string;
  body: string;
}

export function ReportPanel({ content }: Props) {
  const sections = parseSections(content);
  const [collapsed, setCollapsed] = useState<Record<number, boolean>>({});

  const toggleSection = (idx: number) => {
    setCollapsed((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  if (sections.length === 0) {
    return (
      <div className="p-6 overflow-y-auto h-full">
        <div className="leading-relaxed text-warm-text prose prose-sm max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSanitize]}>{content}</ReactMarkdown>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 overflow-y-auto h-full flex flex-col gap-2">
      {sections.map((section, idx) => (
        <div key={idx}>
          <button className="flex items-center gap-2 w-full px-4 py-3 bg-warm-card shadow-sm rounded-xl text-warm-text font-medium text-[15px] text-left hover:bg-warm-sidebar transition-colors duration-150" onClick={() => toggleSection(idx)} aria-expanded={!collapsed[idx]} aria-controls={`section-body-${idx}`}>
            <span className="text-warm-text-secondary text-xs">{collapsed[idx] ? "\u25B6" : "\u25BC"}</span>
            {section.title}
          </button>
          {!collapsed[idx] && (
            <div className="px-4 py-3 leading-relaxed text-warm-text prose prose-sm max-w-none" id={`section-body-${idx}`}>
              <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSanitize]}>{section.body}</ReactMarkdown>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function parseSections(content: string): Section[] {
  const lines = content.split("\n");
  const sections: Section[] = [];
  let currentTitle = "";
  let currentBody: string[] = [];
  for (const line of lines) {
    const headingMatch = line.match(/^#{1,3}\s+(.+)$/);
    if (headingMatch) {
      if (currentTitle) sections.push({ title: currentTitle, body: currentBody.join("\n") });
      currentTitle = headingMatch[1];
      currentBody = [];
    } else {
      currentBody.push(line);
    }
  }
  if (currentTitle) sections.push({ title: currentTitle, body: currentBody.join("\n") });
  return sections;
}
