// frontend/idf-editor/app/page.tsx
"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload } from "lucide-react"

export default function FileUpload() {
    const [file, setFile] = React.useState<File | null>(null)
    const [uploadStatus, setUploadStatus] = React.useState<"idle" | "uploading" | "success" | "error">("idle")
    const [idfId, setIdfId] = React.useState<string | null>(null)

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0])
        }
    }

    const handleUploadClick = async () => {
        if (!file) {
            alert("Please select a file first")
            return
        }

        setUploadStatus("uploading")

        const formData = new FormData()
        formData.append("file", file)

        try {
            const response = await fetch("/api/idf/upload", {
                method: "POST",
                body: formData
            })

            if (response.ok) { // 检查响应状态码是否为 200-299 范围
                const responseData = await response.json(); // 解析 JSON 响应体
                console.log("File uploaded successfully:", responseData); // 成功上传，打印后端返回的数据
                setIdfId(responseData.idf_id)
                setUploadStatus("success"); // 设置上传状态为 "成功"
                alert(`File "${file.name}" uploaded successfully! IDF ID: ${responseData.idf_id}`); // 提示用户上传成功，并显示 IDF ID
            } else {
                const errorData = await response.json(); // 解析 JSON 错误响应体
                console.error("File upload failed:", errorData); // 打印错误信息
                setUploadStatus("error") // 设置上传状态为 "错误"
                alert(`File upload failed: ${errorData.message || "Unknown error"}`) // 提示用户上传失败，显示错误信息
            }
        } catch (error) {
            console.error("Fetch error:", error); // 捕获网络或请求错误
            setUploadStatus("error"); // 设置上传状态为 "错误"
            alert("File upload failed due to a network error."); // 提示用户网络错误
        }
    };

    const handleGeometryRequest = async () => {
        if (!idfId) {
            alert("Please upload a file first")
            return
        }
    }

    const getButtonText = () => { // 根据上传状态动态显示按钮文字
        switch (uploadStatus) {
            case "uploading":
                return "Uploading...";
            case "success":
                return "Success!";
            case "error":
                return "Upload Failed";
            default:
                return "Upload";
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-pink-50">
            <Card className="w-[300px] shadow-lg">
                <CardHeader className="text-center">
                    <CardTitle className="text-2xl font-bold">IDF-Editor</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg p-6 hover:border-gray-400 transition-colors">
                        <input type="file" id="file-upload" className="hidden" onChange={handleFileChange} accept=".idf" />
                        <label htmlFor="file-upload" className="flex flex-col items-center cursor-pointer">
                            <Upload className="h-8 w-8 text-gray-500 mb-2" />
                            <span className="text-sm text-gray-600">{file ? file.name : "Click to upload IDF File"}</span>
                        </label>
                    </div>
                    <Button 
                        className="w-full bg-gray-100 hover:bg-gray-200 text-gray-900"
                        onClick={handleUploadClick}
                        disabled={uploadStatus === "uploading"}>{getButtonText()}</Button>
                </CardContent>
            </Card>
        </div>
    )
}

