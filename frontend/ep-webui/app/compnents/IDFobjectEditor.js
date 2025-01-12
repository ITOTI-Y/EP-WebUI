import { useState } from "react";

function SingleIDFObjectEditor({ obj, onChange }) {
    function handleValueChange(e) {
        const newObj = {...obj, value: e.target.value};
        onChange(newObj);
    }

    function handleProgramlineChange(i, newVal) {
        const newProgramline = [...obj.programline];
        newProgramline[i] = newVal;
        const newObj = {...obj, programline: newProgramline};
        onChange(newObj);
    }

    function renderTableRows() {
        const { note = [], programline = [], units = [] } = obj;

        return note.map((noteText, i) => (
            <tr key={i}>
                {/* Note */}
                <td style={tdStyle}>{noteText}</td>
                {/* Programline */}
                <td style={tdStyle}>
                    <input
                        type="text"
                        value={programline[i] || ""}
                        onChange={(e) => handleProgramlineChange(i, e.target.value)}
                        style={{ width: "100%"}}
                    />
                </td>
                {/* Units */}
                <td style={tdStyle}>{units[i] || ""}</td>
            </tr>
        ));
    }

    return (
        <div style={{ border: "1px solid #ccc", padding: 12, marginBottom: 12 }}>
        <div style={{ marginBottom: 8 }}>
        <label>Type: </label>
        <span style={{ marginRight: 20 }}>{obj.type || ""}</span>

        {obj.value && (
            <>
            <label>Value: </label>
            <input
                type="text"
                value={obj.value}
                onChange={handleValueChange}
                style={{ marginRight: 20 }}
            />
            </>
        )}

        {obj.name !== undefined && (
            <>
            <label>Name: </label>
            <span style={{ marginRight: 20 }}>{obj.name || ""}</span>
            </>
        )}
        </div>

        {renderTableRows().length > 0 && (
            <table style={{ borderCollapse: "collapse", width: "100%" }}>
            <thead>
                <tr>
                <th style={thStyle}>Note</th>
                <th style={thStyle}>Value</th>
                <th style={thStyle}>Unit</th>
                </tr>
            </thead>
            <tbody>{renderTableRows()}</tbody>
            </table>
        )}
    </div>
    )
}

export default function IDFobjectEditor({ objects }) {
    const [idfArray, setIdfArray] = useState(objects);

    function handleSingleObjectChange(i, updateObj) {
        const newArray = [...idfArray];
        newArray[i] = updateObj;
        setIdfArray(newArray);
    }

    return (
        <div style={{ margin: 20 }}>
        <h2>IDF Editor (Array of Objects)</h2>

        {idfArray.map((obj, i) => (
        <SingleIDFObjectEditor
            key={i}
            obj={obj}
            onChange={(updated) => handleSingleObjectChange(i, updated)}
        />
        ))}

      {/* Can add a button to save the idfArray */}
    </div>
    )
}

const thStyle = {
    border: "1px solid #ccc",
    background: "#f9f9f9",
    padding: 4
    };
    
const tdStyle = {
    border: "1px solid #ccc",
    padding: 4
    };