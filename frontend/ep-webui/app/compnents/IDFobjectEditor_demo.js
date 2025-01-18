import { useState } from "react";

export default function IDFobjectEditor({ objects }) {
  const [idfArray, setIdfArray] = useState(objects);

  // 1) 根据 type 分组
  const groupedByType = idfArray.reduce((acc, obj) => {
    const objType = obj.type || "Unknown";
    if (!acc[objType]) {
      acc[objType] = [];
    }
    acc[objType].push(obj);
    return acc;
  }, {});

  return (
    <div style={{ margin: 20 }}>
      <h2>IDF Editor (Grouped by Type)</h2>

      {Object.keys(groupedByType).map((typeKey) => {
        const group = groupedByType[typeKey];

        // 按照是否存在 programline 再次区分
        const withProgramline = group.filter(
          (o) => o.programline && o.programline.length > 0
        );
        const withoutProgramline = group.filter(
          (o) => !o.programline || o.programline.length === 0
        );

        return (
          <div key={typeKey} style={{ border: "1px solid #ccc", marginBottom: 12 }}>
            {/* 区块标题：Type */}
            <div style={{ background: "#f2f2f2", padding: 8 }}>
              <strong>Type: {typeKey}</strong>
            </div>

            {/* ================ 2) 有 programline 的表格 ================ */}
            {withProgramline.length > 0 && (
              <table style={{ borderCollapse: "collapse", width: "100%" }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Name</th>
                    <th style={thStyle}>Value</th>
                    <th style={thStyle}>Note</th>
                    <th style={thStyle}>Programline</th>
                    <th style={thStyle}>Unit</th>
                  </tr>
                </thead>
                <tbody>
                  {withProgramline.map((obj, objIndex) => {
                    // note、programline、units 可能长度不一致，取最大值遍历
                    const rowCount = Math.max(
                      obj.note?.length || 0,
                      obj.programline?.length || 0,
                      obj.units?.length || 0
                    );
                    return Array.from({ length: rowCount }).map((_, rowIndex) => {
                      return (
                        <tr key={`${objIndex}-${rowIndex}`}>
                          {/* 仅在第一行显示 Name、Value，其它行合并单元格 */}
                          {rowIndex === 0 && (
                            <>
                              <td style={tdStyle} rowSpan={rowCount}>
                                {obj.name || ""}
                              </td>
                              <td style={tdStyle} rowSpan={rowCount}>
                                {obj.value || ""}
                              </td>
                            </>
                          )}

                          <td style={tdStyle}>{obj.note?.[rowIndex] || ""}</td>
                          <td style={tdStyle}>
                            {obj.programline?.[rowIndex] || ""}
                          </td>
                          <td style={tdStyle}>{obj.units?.[rowIndex] || ""}</td>
                        </tr>
                      );
                    });
                  })}
                </tbody>
              </table>
            )}

            {/* ================ 3) 无 programline 的表格 ================ */}
            {withoutProgramline.length > 0 && (
              <table style={{ borderCollapse: "collapse", width: "100%", marginTop: 16 }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Type</th>
                    <th style={thStyle}>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {withoutProgramline.map((obj, idx) => (
                    <tr key={idx}>
                      <td style={tdStyle}>{typeKey}</td>
                      <td style={tdStyle}>{obj.value || ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        );
      })}
    </div>
  );
}

// 这里为了示例简单直接写了 style，你也可以挪到全局或别的地方
const thStyle = {
  border: "1px solid #ccc",
  background: "#f9f9f9",
  padding: 4,
};

const tdStyle = {
  border: "1px solid #ccc",
  padding: 4,
};
