import { useState } from "react";

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
      <div className="report-panel">
        <div
          className="report-body"
          dangerouslySetInnerHTML={{ __html: formatMarkdown(content) }}
        />
      </div>
    );
  }

  return (
    <div className="report-panel">
      {sections.map((section, idx) => (
        <div key={idx} className="report-section">
          <button
            className="section-header"
            onClick={() => toggleSection(idx)}
          >
            <span className="collapse-icon">
              {collapsed[idx] ? "▶" : "▼"}
            </span>
            {section.title}
          </button>
          {!collapsed[idx] && (
            <div
              className="section-body"
              dangerouslySetInnerHTML={{ __html: formatMarkdown(section.body) }}
            />
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
      if (currentTitle) {
        sections.push({ title: currentTitle, body: currentBody.join("\n") });
      }
      currentTitle = headingMatch[1];
      currentBody = [];
    } else {
      currentBody.push(line);
    }
  }
  if (currentTitle) {
    sections.push({ title: currentTitle, body: currentBody.join("\n") });
  }

  return sections;
}

function formatMarkdown(text: string): string {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[Source: (.+?)\]/g, '<span class="citation">[Source: $1]</span>')
    .replace(/^- (.+)$/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>)/gs, "<ul>$1</ul>")
    .replace(/\n/g, "<br />");
}
