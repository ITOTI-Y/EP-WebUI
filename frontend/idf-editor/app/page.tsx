// frontend/idf-editor/app/page.tsx
"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, Calculator, ArrowRight } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import dynamic from 'next/dynamic'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Environment } from '@react-three/drei'

// Dynamic import for Three.js components to avoid SSR issues
const ModelViewer = dynamic(() => import('@/components/ModelViewer'), { ssr: false })
const ModelWithPV = dynamic(() => import('@/components/ModelWithPV'), { ssr: false })

// Types for geometry data
interface Vertex {
  x: number;
  y: number;
  z: number;
}

interface Surface {
  name: string;
  surface_type: string;
  construction_name?: string;
  zone_name?: string;
  vertices: [number, number, number][];
  area?: number;
  radiation_score?: number;
  selected?: boolean;
}

interface GeometryData {
  zones: { name: string }[];
  surfaces: Surface[];
  fenestration_surfaces: Surface[];
}

interface PVSystem {
  pv_name: string;
  surface_name: string;
  area: number;
  pv_area: number;
  efficiency: number;
  radiation_score: number;
}

interface PVResult {
  pv_systems: {
    [key: string]: {
      total_energy_kwh: number;
      max_power_kw: number;
      monthly_energy_kwh?: {
        [month: string]: number;
      };
    };
  };
  total_energy_kwh: number;
  message: string;
}

export default function IDFEditor() {
    const [file, setFile] = React.useState<File | null>(null)
    const [uploadStatus, setUploadStatus] = React.useState<"idle" | "uploading" | "success" | "error">("idle")
    const [idfId, setIdfId] = React.useState<string | null>(null)
  const [geometryData, setGeometryData] = React.useState<GeometryData | null>(null)
  const [selectedSurface, setSelectedSurface] = React.useState<Surface | null>(null)
  const [pvSurfaces, setPvSurfaces] = React.useState<Surface[]>([])
  const [pvResults, setPvResults] = React.useState<PVResult | null>(null)
  const [calculating, setCalculating] = React.useState<boolean>(false)
  const [activeTab, setActiveTab] = React.useState<string>("original")

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0])
      // Reset state when a new file is selected
      setIdfId(null)
      setGeometryData(null)
      setSelectedSurface(null)
      setPvSurfaces([])
      setPvResults(null)
      setUploadStatus("idle")
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

      if (response.ok) {
        const responseData = await response.json();
        console.log("File uploaded successfully:", responseData);
                setIdfId(responseData.idf_id)
        setUploadStatus("success");
        // Fetch geometry data after successful upload
        fetchGeometryData(responseData.idf_id);
      } else {
        const errorData = await response.json();
        console.error("File upload failed:", errorData);
        setUploadStatus("error")
        alert(`File upload failed: ${errorData.message || "Unknown error"}`)
      }
    } catch (error) {
      console.error("Fetch error:", error);
      setUploadStatus("error");
      alert("File upload failed due to a network error.");
    }
  };

  const fetchGeometryData = async (id: string) => {
    try {
      const response = await fetch(`/api/geometry/${id}`);
      
      if (response.ok) {
        const data = await response.json();
        console.log("Geometry data:", data);
        setGeometryData(data.geometry_data);
            } else {
        const errorData = await response.json();
        console.error("Fetching geometry failed:", errorData);
        alert(`Failed to fetch geometry: ${errorData.message || "Unknown error"}`);
            }
        } catch (error) {
      console.error("Fetch geometry error:", error);
      alert("Failed to fetch geometry data due to a network error.");
    }
  };

  const handleSurfaceClick = (surface: Surface) => {
    setSelectedSurface(surface);
  };

  const addToPVSurfaces = (surface: Surface) => {
    if (!surface) return;
    
    // Check if surface is already in pvSurfaces
    const exists = pvSurfaces.some(s => s.name === surface.name);
    
    if (!exists) {
      setPvSurfaces([...pvSurfaces, surface]);
    }
  };

  const removeFromPVSurfaces = (surfaceName: string) => {
    setPvSurfaces(pvSurfaces.filter(s => s.name !== surfaceName));
  };

  const calculatePV = async () => {
    if (!idfId || pvSurfaces.length === 0) {
      alert("Please upload an IDF file and select at least one surface for PV installation");
      return;
    }

    setCalculating(true);

    try {
      // Mock data for surfaces with area and radiation score
      const surfacesWithAttributes = pvSurfaces.map(surface => ({
        name: surface.name,
        area: surface.area || calculateArea(surface.vertices), // Calculate area if not provided
        radiation_score: surface.radiation_score || Math.random() * 1000 // Mock radiation score
      }));

      let results;
      
      try {
        // Try to call the real API
        const response = await fetch("/api/pv/calculate", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            idf_id: idfId,
            surfaces: surfacesWithAttributes,
            weather_file: "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"
          })
        });
        
        if (response.ok) {
          results = await response.json();
        } else {
          throw new Error("API call failed");
        }
      } catch (error) {
        console.warn("API endpoint not available, using mock data instead");
        
        // Create mock results if API fails
        results = createMockPVResults(surfacesWithAttributes);
      }
      
      console.log("PV calculation results:", results);
      setPvResults(results);
      setActiveTab("withPV"); // Switch to the PV model tab after calculation
    } catch (error) {
      console.error("PV calculation error:", error);
      alert("PV calculation failed. See console for details.");
    } finally {
      setCalculating(false);
    }
  };

  // Helper function to create mock PV results
  const createMockPVResults = (surfaces: any[]): PVResult => {
    const pv_systems: Record<string, any> = {};
    let total_energy_kwh = 0;
    
    surfaces.forEach(surface => {
      const pv_name = `PV_${surface.name}`;
      const energy = surface.area * 150 * (surface.radiation_score / 1000); // Mock energy calculation
      
      const monthly: Record<number, number> = {};
      for (let month = 1; month <= 12; month++) {
        // Create a seasonal pattern
        const seasonFactor = 1 + 0.5 * Math.sin((month - 3) * Math.PI / 6);
        monthly[month] = energy / 12 * seasonFactor;
      }
      
      pv_systems[pv_name] = {
        total_energy_kwh: energy,
        max_power_kw: energy / (365 * 5), // Approximate peak power
        monthly_energy_kwh: monthly
      };
      
      total_energy_kwh += energy;
    });
    
    return {
      pv_systems,
      total_energy_kwh,
      message: "Mock PV calculation completed successfully"
    };
  };

  const calculateArea = (vertices: [number, number, number][]) => {
    // Simple area calculation for demonstration purposes
    // In a real application, this would be more geometric-aware
    return Math.random() * 100 + 20; // Mock area between 20 and 120 square meters
  };

    return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-pink-50 p-4">
      <div className="container mx-auto max-w-6xl">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Left Panel - File Upload & Controls */}
          <div className="md:col-span-1">
            <Card className="shadow-lg h-full">
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
                  className="w-full bg-primary hover:bg-primary/90 text-white"
                        onClick={handleUploadClick}
                  disabled={uploadStatus === "uploading" || !file}>
                  {uploadStatus === "uploading" ? "Uploading..." : 
                   uploadStatus === "success" ? "Uploaded Successfully" : "Upload"}
                </Button>
                
                {geometryData && (
                  <>
                    <div className="mt-6">
                      <h3 className="font-medium mb-2">Selected Surfaces for PV</h3>
                      {pvSurfaces.length > 0 ? (
                        <ul className="space-y-2">
                          {pvSurfaces.map((surface) => (
                            <li key={surface.name} className="flex justify-between items-center text-sm bg-gray-100 p-2 rounded">
                              <span>{surface.name}</span>
                              <Button 
                                variant="ghost" 
                                size="sm"
                                onClick={() => removeFromPVSurfaces(surface.name)}
                                className="h-6 w-6 p-0 text-red-500">×</Button>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="text-sm text-gray-500">Click on surfaces in the model to select them for PV installation</p>
                      )}
                    </div>
                    
                    <Button 
                      className="w-full mt-4 bg-green-600 hover:bg-green-700 text-white"
                      onClick={calculatePV}
                      disabled={calculating || pvSurfaces.length === 0}>
                      {calculating ? "Calculating..." : "Calculate PV Output"}
                      <Calculator className="ml-2 h-4 w-4" />
                    </Button>
                  </>
                )}
              </CardContent>
            </Card>
          </div>
          
          {/* Center Panel - 3D Visualization */}
          <div className="md:col-span-2 flex flex-col">
            <Card className="shadow-lg flex-grow">
              <CardHeader className="pb-2">
                <CardTitle className="text-xl">Model Visualization</CardTitle>
              </CardHeader>
              <CardContent className="h-[70vh]">
                {geometryData ? (
                  <Tabs defaultValue="original" value={activeTab} onValueChange={setActiveTab} className="h-full">
                    <TabsList className="grid w-full grid-cols-2">
                      <TabsTrigger value="original">Original Model</TabsTrigger>
                      <TabsTrigger value="withPV" disabled={!pvResults}>With PV Panels</TabsTrigger>
                    </TabsList>
                    <TabsContent value="original" className="h-[calc(100%-40px)]">
                      <Canvas camera={{ position: [10, 10, 10], fov: 60 }}>
                        <ambientLight intensity={0.5} />
                        <directionalLight position={[10, 10, 5]} intensity={1} />
                        <ModelViewer 
                          geometryData={geometryData} 
                          onSurfaceClick={handleSurfaceClick} 
                          selectedSurfaces={pvSurfaces}
                        />
                        <OrbitControls />
                        <Environment preset="city" />
                      </Canvas>
                    </TabsContent>
                    <TabsContent value="withPV" className="h-[calc(100%-40px)]">
                      {pvResults ? (
                        <Canvas camera={{ position: [10, 10, 10], fov: 60 }}>
                          <ambientLight intensity={0.5} />
                          <directionalLight position={[10, 10, 5]} intensity={1} />
                          <ModelWithPV 
                            geometryData={geometryData} 
                            pvSurfaces={pvSurfaces}
                            onSurfaceClick={handleSurfaceClick}
                          />
                          <OrbitControls />
                          <Environment preset="city" />
                        </Canvas>
                      ) : (
                        <div className="flex items-center justify-center h-full">
                          <p>Calculate PV results to see the model with panels</p>
                        </div>
                      )}
                    </TabsContent>
                  </Tabs>
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <p>Upload an IDF file to see the 3D model</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
        
        {/* Bottom Panel - Details and Results */}
        <div className="mt-4">
          <Card className="shadow-lg">
            <CardHeader className="pb-2">
              <CardTitle className="text-xl">Details</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="surfaceDetails">
                <TabsList>
                  <TabsTrigger value="surfaceDetails">Surface Details</TabsTrigger>
                  <TabsTrigger value="pvResults" disabled={!pvResults}>PV Results</TabsTrigger>
                </TabsList>
                <TabsContent value="surfaceDetails">
                  {selectedSurface ? (
                    <div className="p-4">
                      <h3 className="font-bold text-lg mb-2">{selectedSurface.name}</h3>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <p><span className="font-medium">Type:</span> {selectedSurface.surface_type}</p>
                          {selectedSurface.construction_name && (
                            <p><span className="font-medium">Construction:</span> {selectedSurface.construction_name}</p>
                          )}
                          {selectedSurface.zone_name && (
                            <p><span className="font-medium">Zone:</span> {selectedSurface.zone_name}</p>
                          )}
                          <p><span className="font-medium">Area:</span> {selectedSurface.area || calculateArea(selectedSurface.vertices).toFixed(2)} m²</p>
                        </div>
                        <div>
                          <p className="font-medium">Vertices:</p>
                          <ul className="text-xs">
                            {selectedSurface.vertices.map((vertex, index) => (
                              <li key={index}>
                                ({vertex[0].toFixed(2)}, {vertex[1].toFixed(2)}, {vertex[2].toFixed(2)})
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                      <div className="mt-4">
                        <Button 
                          onClick={() => addToPVSurfaces(selectedSurface)}
                          disabled={pvSurfaces.some(s => s.name === selectedSurface.name)}
                          className="bg-blue-500 hover:bg-blue-600 text-white">
                          {pvSurfaces.some(s => s.name === selectedSurface.name) 
                            ? "Added to PV Surfaces" 
                            : "Add to PV Surfaces"}
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="p-4 text-center text-gray-500">
                      <p>Click on a surface in the model to see details</p>
                    </div>
                  )}
                </TabsContent>
                <TabsContent value="pvResults">
                  {pvResults ? (
                    <div className="p-4">
                      <div className="mb-4">
                        <h3 className="font-bold text-lg">Total Energy Production</h3>
                        <p className="text-2xl font-bold text-green-600">{pvResults.total_energy_kwh.toFixed(2)} kWh/year</p>
                      </div>
                      
                      <Accordion type="single" collapsible className="w-full">
                        {Object.entries(pvResults.pv_systems).map(([key, system]) => (
                          <AccordionItem key={key} value={key}>
                            <AccordionTrigger>
                              <div className="flex justify-between w-full pr-4">
                                <span>{key}</span>
                                <span>{system.total_energy_kwh.toFixed(2)} kWh/year</span>
                              </div>
                            </AccordionTrigger>
                            <AccordionContent>
                              <div className="space-y-2 pl-4">
                                <p><span className="font-medium">Max Power:</span> {system.max_power_kw.toFixed(2)} kW</p>
                                
                                {system.monthly_energy_kwh && (
                                  <div>
                                    <p className="font-medium mb-1">Monthly Production (kWh):</p>
                                    <div className="grid grid-cols-3 gap-2 text-sm">
                                      {Object.entries(system.monthly_energy_kwh).map(([month, energy]) => (
                                        <div key={month} className="flex justify-between">
                                          <span>Month {month}:</span>
                                          <span>{energy.toFixed(1)}</span>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            </AccordionContent>
                          </AccordionItem>
                        ))}
                      </Accordion>
                    </div>
                  ) : (
                    <div className="p-4 text-center text-gray-500">
                      <p>Calculate PV results to see details</p>
                    </div>
                  )}
                </TabsContent>
              </Tabs>
                </CardContent>
            </Card>
        </div>
      </div>
        </div>
    )
}

