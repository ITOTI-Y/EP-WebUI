import { useState } from "react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./table";
import { Listbox, ListboxButton, ListboxLabel, ListboxOption, ListboxOptions } from "./listbox";
import { useGetFilename, useGetIDFData } from "../hooks/usedata";
import SaveButton from "./savebutton";

export default function IDFobjectEditor() {
    const [filenames, error] = useGetFilename();
    const [selectedFilename, setSelectedFilename] = useState(null);
    const [idfData, errorIDFData] = useGetIDFData(selectedFilename);

    if (error) {
        console.log("获取文件列表时出现错误：", error);
    }
    if (errorIDFData) {
        console.log("获取 IDF 数据时出现错误：", errorIDFData);
    }

    function handleFilenameChange(filenameObj) {
        // 当用户在下拉列表中选择了某个文件时，更新 selectedFilename
        setSelectedFilename(filenameObj.name);
    }

    return (
        <div>
            <div className="flex items-end justify-between gap-4">
                <div className="w-1/3">
                    <label htmlFor="filename" className="py-2 text-sm font-medium text-gray-700">
                        Filename
                    </label>
                    <div>
                        <Listbox onChange={handleFilenameChange}>
                            {filenames && filenames.map(nameObj => (
                                <ListboxOption key={nameObj.id} value={nameObj}>
                                    <ListboxLabel>{nameObj.name}</ListboxLabel>
                                </ListboxOption>
                            ))}
                        </Listbox>
                    </div>
                </div>
                <SaveButton />
            </div>

            {/* 可以根据 idfData 在此处渲染相应的内容 */}
            <Table>
                <TableHead>
                    <TableRow>
                        <TableHeader>Type</TableHeader>
                        <TableHeader>Unit</TableHeader>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {idfData && idfData.map((item, index) => (
                        <TableRow key={index}>
                            <TableCell>{item.type}</TableCell>
                            <TableCell>{item.unit}</TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </div>
    );
}
