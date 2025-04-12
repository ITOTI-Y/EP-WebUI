import { useState, useEffect } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'

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
  selected?: boolean;
}

interface GeometryData {
  zones: { name: string }[];
  surfaces: Surface[];
  fenestration_surfaces: Surface[];
}

interface ModelViewerProps {
  geometryData: GeometryData;
  onSurfaceClick: (surface: Surface) => void;
  selectedSurfaces: Surface[];
}

export default function ModelViewer({ geometryData, onSurfaceClick, selectedSurfaces }: ModelViewerProps) {
  const { camera } = useThree();
  const [hoveredSurface, setHoveredSurface] = useState<string | null>(null);

  // Helper to get material based on surface type and selection state
  const getMaterial = (surface: Surface) => {
    const isSelected = selectedSurfaces.some(s => s.name === surface.name);
    const isHovered = hoveredSurface === surface.name;
    
    // Base opacity and color
    let opacity = isHovered ? 0.9 : 0.8;
    let color;
    
    // Determine color based on surface type
    switch (surface.surface_type.toUpperCase()) {
      case 'WALL':
        color = new THREE.Color('#B3C2D1');
        break;
      case 'ROOF':
      case 'CEILING':
        color = new THREE.Color('#7B9EB3');
        break;
      case 'FLOOR':
        color = new THREE.Color('#94A7BC');
        break;
      default:
        color = new THREE.Color('#CCCCCC');
    }
    
    // Highlight selected surfaces with a green tint
    if (isSelected) {
      color.lerp(new THREE.Color('#4CAF50'), 0.5);
      opacity = 0.95;
    }
    
    return new THREE.MeshStandardMaterial({ 
      color,
      opacity,
      transparent: true,
      side: THREE.DoubleSide,
      metalness: 0.1,
      roughness: 0.7,
    });
  };

  // Calculate center of model for camera positioning
  const calculateCenter = () => {
    const allVertices = geometryData.surfaces.flatMap(s => s.vertices);
    if (allVertices.length === 0) return [0, 0, 0];
    
    const sum = allVertices.reduce(
      (acc, vertex) => [acc[0] + vertex[0], acc[1] + vertex[1], acc[2] + vertex[2]],
      [0, 0, 0]
    );
    
    return [
      sum[0] / allVertices.length,
      sum[1] / allVertices.length,
      sum[2] / allVertices.length
    ];
  };

  // Adjust camera on first render
  useEffect(() => {
    const center = calculateCenter();
    camera.lookAt(new THREE.Vector3(center[0], center[1], center[2]));
    
    // Find max dimension to set camera distance
    const allVertices = geometryData.surfaces.flatMap(s => s.vertices);
    const distances = allVertices.map(v => 
      Math.sqrt(
        Math.pow(v[0] - center[0], 2) + 
        Math.pow(v[1] - center[1], 2) + 
        Math.pow(v[2] - center[2], 2)
      )
    );
    
    const maxDistance = Math.max(...distances);
    camera.position.set(
      center[0] + maxDistance * 1.5,
      center[1] + maxDistance * 1.5,
      center[2] + maxDistance * 1.5
    );
    
    camera.updateProjectionMatrix();
  }, [geometryData, camera]);

  // Improved surface mesh creation
  const createSurfaceMesh = (surface: Surface, material: THREE.Material) => {
    const vertices = surface.vertices;
    
    if (vertices.length < 3) return null;
    
    // Create geometry directly using BufferGeometry
    const geometry = new THREE.BufferGeometry();
    
    // Convert vertices to the format expected by Three.js
    const positions: number[] = [];
    const indices: number[] = [];
    
    // Add all vertices
    vertices.forEach(vertex => {
      positions.push(vertex[0], vertex[1], vertex[2]);
    });
    
    // Create triangles (faces)
    // We create a simple fan triangulation - works for convex polygons
    for (let i = 1; i < vertices.length - 1; i++) {
      indices.push(0, i, i + 1);
    }
    
    geometry.setIndex(indices);
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.computeVertexNormals();
    
    return (
      <mesh
        geometry={geometry}
        material={material}
        onClick={() => onSurfaceClick(surface)}
        onPointerOver={() => setHoveredSurface(surface.name)}
        onPointerOut={() => setHoveredSurface(null)}
      />
    );
  };

  return (
    <group>
      {/* Render regular surfaces */}
      {geometryData.surfaces.map((surface, index) => {
        return createSurfaceMesh(surface, getMaterial(surface));
      })}
      
      {/* Render fenestration surfaces (windows, doors) */}
      {geometryData.fenestration_surfaces.map((surface, index) => {
        // Similar shape creation as above, but simplified for brevity
        const shape = new THREE.Shape();
        const vertices = surface.vertices;
        
        if (vertices.length < 3) return null;
        
        shape.moveTo(vertices[0][0], vertices[0][1]);
        for (let i = 1; i < vertices.length; i++) {
          shape.lineTo(vertices[i][0], vertices[i][1]);
        }
        
        const geometry = new THREE.ShapeGeometry(shape);
        const material = new THREE.MeshStandardMaterial({
          color: '#87CEEB',
          transparent: true,
          opacity: 0.6,
          side: THREE.DoubleSide,
          metalness: 0.2,
          roughness: 0.1,
        });
        
        // Adjust positions similar to surfaces above
        const positions = geometry.attributes.position;
        for (let i = 0; i < positions.count; i++) {
          const x = positions.getX(i);
          const y = positions.getY(i);
          
          // Find matching vertex
          for (const vertex of vertices) {
            if (Math.abs(x - vertex[0]) < 0.001 && Math.abs(y - vertex[1]) < 0.001) {
              positions.setXYZ(i, vertex[0], vertex[1], vertex[2]);
              break;
            }
          }
        }
        
        positions.needsUpdate = true;
        geometry.computeVertexNormals();
        
        return (
          <mesh
            key={`fenestration-${index}`}
            geometry={geometry}
            material={material}
          />
        );
      })}
    </group>
  );
} 