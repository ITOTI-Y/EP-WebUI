"use client"

import { useEffect, useState } from "react";
import IDFobjectEditor from "./compnents/IDFobjectEditor";

export default function Home() {
    // Save filenames from the backend
    const [filenames, setFilenames] = useState([]);

    // The filename that the user selected from dropdown menu
    const [selectedFile, setSelectedFile] = useState("");

    // Use to display and edit the IDF file
    const [idfData, setIdfData] = useState(null);

    // Is loading
    const [loading, setLoading] = useState(false);

    // Is error
    const [error, setErrorMsg] = useState("");

    // Call /api/idf-files to get the list of filenames
    useEffect(() => {
        async function fetchFilenames() {
            try {
                const res = await fetch("http://localhost:8000/api/idf-files");
                if (!res.ok) {
                    throw new Error("Failed to fetch filenames");
                }
                const data = await res.json();
                setFilenames(data.filenames);
            } catch (error) {
                setErrorMsg(error.message);
            }
        }
        fetchFilenames();
    }, []); // Empty array means this effect runs only once when the component mounts

    // When click "Import" button , fetch the specific filename idf file
    async function handleImport() {
        if (!selectedFile) {
            alert("Please select a file from the dropdown menu");
            return;
        }
        setLoading(true);
        setErrorMsg("");

        try {
            const res = await fetch(`
                http://localhost:8000/api/idf-files/${encodeURIComponent(selectedFile)}`
            );
            if (!res.ok) {
                throw new Error(`Failed to import IDF for file: ${selectedFile}`);
            }
            const data = await res.json();
            // data => { filename: "XXX", objects: [...]}
            setIdfData(data.objects);
        } catch (error) {
            setErrorMsg(error.message);
        } finally {
            setLoading(false);
        }
    }

    return (
        <div style={{ margin: "20px" }}>
            <h1>EnergyPlus Templates Import (Next.js)</h1>

            {/* Display the error message (if any) */}
            {error && <p style={{ color: "red" }}>Error: {error}</p>}
            
            {/* Display the dropdown menu */}
            <div sytle={{ marginBottom:10 }}>
                <label>Select IDF File:</label>
                <select 
                    value={selectedFile}
                    onChange={(e) => setSelectedFile(e.target.value)}
                    style={{ width: 200}}
                >
                    <option value="">--Select a file--</option>
                    {filenames.map((fname) => (
                        <option key={fname} value={fname}>
                            {fname}
                        </option>
                    ))}
                </select>

                <button onClick={handleImport} style={{ marginLeft: 10}} disabled={!selectedFile || loading}>
                    {loading ? "Importing..." : "Import"}
                </button>
            </div>

            {/* Display the IDF file (if any) */}
            {idfData && (
                <div style={{ margin: 20 }}>
                    <IDFobjectEditor objects={idfData} />
                </div>
            )}
        </div>
    )
}