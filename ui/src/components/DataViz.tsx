import { useState, useMemo } from "react";
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import type { TableData } from "../types";

interface Props {
  data: TableData;
}

const COLORS = ["#8B7355", "#7C9082", "#B8926A", "#C4898A", "#6B8F9E"];

export function DataViz({ data }: Props) {
  const [view, setView] = useState<"chart" | "table">(
    data.chart && data.chart.type !== "none" ? "chart" : "table"
  );

  const chartData = useMemo(() => {
    if (!data.chart || data.chart.type === "none") return [];
    const xIdx = data.columns.findIndex((c) => c.name === data.chart!.x);
    const yIdxs = data.chart.y.map((name) =>
      data.columns.findIndex((c) => c.name === name)
    );
    if (xIdx === -1) return [];

    return data.rows.map((row) => {
      const point: Record<string, string | number> = { [data.chart!.x]: row[xIdx] };
      for (let i = 0; i < yIdxs.length; i++) {
        if (yIdxs[i] !== -1) {
          const val = parseFloat(row[yIdxs[i]]);
          point[data.chart!.y[i]] = isNaN(val) ? 0 : val;
        }
      }
      return point;
    });
  }, [data]);

  const hasChart = data.chart && data.chart.type !== "none" && chartData.length > 0;

  return (
    <div className="mt-3 border border-warm-border rounded-xl overflow-hidden bg-warm-card">
      {hasChart && (
        <div className="flex border-b border-warm-border bg-warm-sidebar/50">
          <button
            className={`flex-1 px-4 py-2 text-sm transition-colors duration-150 ${
              view === "chart"
                ? "bg-warm-card text-warm-text font-semibold border-b-2 border-warm-accent"
                : "text-warm-text-secondary hover:text-warm-text"
            }`}
            onClick={() => setView("chart")}
          >
            Chart
          </button>
          <button
            className={`flex-1 px-4 py-2 text-sm transition-colors duration-150 ${
              view === "table"
                ? "bg-warm-card text-warm-text font-semibold border-b-2 border-warm-accent"
                : "text-warm-text-secondary hover:text-warm-text"
            }`}
            onClick={() => setView("table")}
          >
            Table
          </button>
        </div>
      )}

      {view === "chart" && hasChart ? (
        <div className="p-4">
          {data.chart!.title && (
            <h4 className="mb-3 text-sm font-medium text-warm-text">{data.chart!.title}</h4>
          )}
          <ResponsiveContainer width="100%" height={300}>
            {data.chart!.type === "line" ? (
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5DDD4" />
                <XAxis dataKey={data.chart!.x} tick={{ fontSize: 12, fill: "#8C7E6F" }} />
                <YAxis tick={{ fontSize: 12, fill: "#8C7E6F" }} />
                <Tooltip contentStyle={{ backgroundColor: "#FAFAF7", border: "1px solid #E5DDD4", borderRadius: "8px", fontSize: "13px" }} />
                <Legend />
                {data.chart!.y.map((col, i) => (
                  <Line key={col} type="monotone" dataKey={col} stroke={COLORS[i % COLORS.length]} dot={chartData.length < 30} />
                ))}
              </LineChart>
            ) : (
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5DDD4" />
                <XAxis dataKey={data.chart!.x} tick={{ fontSize: 12, fill: "#8C7E6F" }} />
                <YAxis tick={{ fontSize: 12, fill: "#8C7E6F" }} />
                <Tooltip contentStyle={{ backgroundColor: "#FAFAF7", border: "1px solid #E5DDD4", borderRadius: "8px", fontSize: "13px" }} />
                <Legend />
                {data.chart!.y.map((col, i) => (
                  <Bar key={col} dataKey={col} fill={COLORS[i % COLORS.length]} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            )}
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr>
                {data.columns.map((col) => (
                  <th key={col.name} className="px-3 py-2 text-left font-semibold text-warm-text-secondary bg-warm-sidebar/50 sticky top-0 border-b border-warm-border">{col.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row, i) => (
                <tr key={i} className="hover:bg-warm-bg transition-colors duration-100">
                  {row.map((cell, j) => (
                    <td key={j} className="px-3 py-2 border-b border-warm-border/50 text-warm-text">{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {data.sql && (
        <details className="px-3 py-2 border-t border-warm-border text-xs text-warm-text-secondary">
          <summary className="cursor-pointer hover:text-warm-text transition-colors duration-150">SQL Query</summary>
          <pre className="mt-2 p-2 bg-warm-sidebar rounded-lg overflow-x-auto text-[11px]">{data.sql}</pre>
        </details>
      )}
    </div>
  );
}
