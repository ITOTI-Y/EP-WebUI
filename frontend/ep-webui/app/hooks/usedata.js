'use client'

import { useState, useEffect } from 'react'
import axios from 'axios'

const apiUrl = process.env.NEXT_PUBLIC_API_URL

export function useGetFilename() {
    const [data, setData] = useState(null)
    const [error, setError] = useState(null)

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get(`${apiUrl}/api/idf-files`)
                const transformedData = response.data.filenames.map((filename, index) => ({
                    id: index,
                    name: filename,
                }))
                setData(transformedData)
            } catch (error) {
                setError(error)
            }
        }
        fetchData()
    }, [])

    // 返回给外部组件使用
    return [data, error]
} 

export function useGetIDFData(filename) {
    const [data, setData] = useState(null)
    const [error, setError] = useState(null)

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get(`${apiUrl}/api/idf-files/${filename}`)
                setData(response.data.objects)
            } catch (error) {
                setError(error)
            }
        }
        fetchData()
    }, [filename])

    return [data, error]
}

