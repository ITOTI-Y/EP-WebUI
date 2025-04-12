// frontend/idf-editor/components/ModelWithPV.tsx
import { useState, useEffect } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'

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

interface ModelWithPVProps {
  geometryData: GeometryData;
  pvSurfaces: Surface[];
  onSurfaceClick: (surface: Surface) => void;
}

export default function ModelWithPV({ geometryData, pvSurfaces, onSurfaceClick }: ModelWithPVProps) {
  const { camera } = useThree();
  const [hoveredSurface, setHoveredSurface] = useState<string | null>(null);

  // Helper to get material based on surface type
  const getMaterial = (surface: Surface) => {
    const isPV = pvSurfaces.some(s => s.name === surface.name);
    const isHovered = hoveredSurface === surface.name;
    
    // Base opacity and color
    let opacity = isHovered ? 0.9 : 0.8;
    let color;
    
    // Determine color based on surface type
    if (isPV) {
      // PV panel appearance
      return new THREE.MeshStandardMaterial({ 
        color: new THREE.Color('#1E3A8A'), // Dark blue for PV panels
        opacity: 0.95,
        transparent: true,
        side: THREE.DoubleSide,
        metalness: 0.8,
        roughness: 0.2,
      });
    }
    
    // Regular surfaces
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
    
    return new THREE.MeshStandardMaterial({ 
      color,
      opacity,
      transparent: true,
      side: THREE.DoubleSide,
      metalness: 0.1,
      roughness: 0.7,
    });
  };

  // For PV panels, create a slightly offset surface with solar cell appearance
  const createPVPanelMesh = (surface: Surface) => {
    const vertices = surface.vertices;
    
    if (vertices.length < 3) return null;
    
    // Calculate normal vector of the surface
    const v1 = new THREE.Vector3(...vertices[0]);
    const v2 = new THREE.Vector3(...vertices[1]);
    const v3 = new THREE.Vector3(...vertices[2]);
    
    const edge1 = new THREE.Vector3().subVectors(v2, v1);
    const edge2 = new THREE.Vector3().subVectors(v3, v1);
    const normal = new THREE.Vector3().crossVectors(edge1, edge2).normalize();
    
    // Create a shape for the panel
    const shape = new THREE.Shape();
    
    // Choose projection plane similar to ModelViewer component
    let xIndex = 0, yIndex = 1;
    const absNormal = normal.clone().set(Math.abs(normal.x), Math.abs(normal.y), Math.abs(normal.z));
    
    if (absNormal.z > absNormal.x && absNormal.z > absNormal.y) {
      xIndex = 0;
      yIndex = 1;
    } else if (absNormal.y > absNormal.x) {
      xIndex = 0;
      yIndex = 2;
    } else {
      xIndex = 1;
      yIndex = 2;
    }
    
    // Create the base shape
    shape.moveTo(vertices[0][xIndex], vertices[0][yIndex]);
    for (let i = 1; i < vertices.length; i++) {
      shape.lineTo(vertices[i][xIndex], vertices[i][yIndex]);
    }
    shape.lineTo(vertices[0][xIndex], vertices[0][yIndex]);
    
    // Create geometry from shape
    const geometry = new THREE.ShapeGeometry(shape);
    
    // Adjust positions to match 3D coordinates
    const positions = geometry.attributes.position;
    const positionArray = positions.array;
    
    for (let i = 0; i < positionArray.length; i += 3) {
      let found = false;
      
      // Try to match with original vertices
      for (const vertex of vertices) {
        const x = vertex[xIndex];
        const y = vertex[yIndex];
        
        if (
          Math.abs(positionArray[i + xIndex % 3] - x) < 0.001 &&
          Math.abs(positionArray[i + yIndex % 3] - y) < 0.001
        ) {
          // Found matching vertex, set the 3D coordinates
          // Offset slightly along normal for visual separation
          positionArray[i] = vertex[0] + normal.x * 0.05;
          positionArray[i + 1] = vertex[1] + normal.y * 0.05;
          positionArray[i + 2] = vertex[2] + normal.z * 0.05;
          found = true;
          break;
        }
      }
      
      // If no match, use the normal to set the depth
      if (!found) {
        positionArray[i + (3 - xIndex - yIndex) % 3] = vertices[0][3 - xIndex - yIndex] + normal.z * 0.05;
      }
    }
    
    positions.needsUpdate = true;
    geometry.computeVertexNormals();
    
    // Create the PV panel material with solar cell appearance
    const pvMaterial = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#1E3A8A'), // Dark blue for PV panels
      opacity: 0.95,
      transparent: true,
      side: THREE.DoubleSide,
      metalness: 0.8,
      roughness: 0.2,
    });
    
    return (
      <mesh
        geometry={geometry}
        material={pvMaterial}
        onClick={() => onSurfaceClick(surface)}
        onPointerOver={() => setHoveredSurface(surface.name)}
        onPointerOut={() => setHoveredSurface(null)}
      />
    );
  };

  // Calculate center of model for camera positioning (same as in ModelViewer)
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

  return (
    <group>
      {/* Render all surfaces */}
      {geometryData.surfaces.map((surface, index) => {
        // Similar shape creation as in ModelViewer
        const shape = new THREE.Shape();
        const vertices = surface.vertices;
        
        if (vertices.length < 3) return null;
        
        // Determine the normal of the surface
        const v1 = new THREE.Vector3(...vertices[0]);
        const v2 = new THREE.Vector3(...vertices[1]);
        const v3 = new THREE.Vector3(...vertices[2]);
        
        const edge1 = new THREE.Vector3().subVectors(v2, v1);
        const edge2 = new THREE.Vector3().subVectors(v3, v1);
        const normal = new THREE.Vector3().crossVectors(edge1, edge2).normalize();
        
        // Choose projection plane based on the normal
        let xIndex = 0, yIndex = 1;
        const absNormal = normal.clone().set(Math.abs(normal.x), Math.abs(normal.y), Math.abs(normal.z));
        
        if (absNormal.z > absNormal.x && absNormal.z > absNormal.y) {
          xIndex = 0;
          yIndex = 1;
        } else if (absNormal.y > absNormal.x) {
          xIndex = 0;
          yIndex = 2;
        } else {
          xIndex = 1;
          yIndex = 2;
        }
        
        // Create the 2D shape
        shape.moveTo(vertices[0][xIndex], vertices[0][yIndex]);
        for (let i = 1; i < vertices.length; i++) {
          shape.lineTo(vertices[i][xIndex], vertices[i][yIndex]);
        }
        shape.lineTo(vertices[0][xIndex], vertices[0][yIndex]);
        
        // Create geometry
        const geometry = new THREE.ShapeGeometry(shape);
        
        // Adjust vertices
        const positions = geometry.attributes.position;
        const positionArray = positions.array;
        
        for (let i = 0; i < positionArray.length; i += 3) {
          let found = false;
          
          for (const vertex of vertices) {
            const x = vertex[xIndex];
            const y = vertex[yIndex];
            
            if (
              Math.abs(positionArray[i + xIndex % 3] - x) < 0.001 &&
              Math.abs(positionArray[i + yIndex % 3] - y) < 0.001
            ) {
              positionArray[i] = vertex[0];
              positionArray[i + 1] = vertex[1];
              positionArray[i + 2] = vertex[2];
              found = true;
              break;
            }
          }
          
          if (!found) {
            positionArray[i + (3 - xIndex - yIndex) % 3] = vertices[0][3 - xIndex - yIndex];
          }
        }
        
        positions.needsUpdate = true;
        geometry.computeVertexNormals();
        
        const isPV = pvSurfaces.some(s => s.name === surface.name);
        
        // For non-PV surfaces, render normally
        if (!isPV) {
          return (
            <mesh
              key={`surface-${index}`}
              geometry={geometry}
              material={getMaterial(surface)}
              onClick={() => onSurfaceClick(surface)}
              onPointerOver={() => setHoveredSurface(surface.name)}
              onPointerOut={() => setHoveredSurface(null)}
            />
          );
        }
        
        // For PV surfaces, render the base surface plus the PV panel on top
        return (
          <group key={`pv-group-${index}`}>
            <mesh
              geometry={geometry}
              material={getMaterial(surface)}
              onClick={() => onSurfaceClick(surface)}
              onPointerOver={() => setHoveredSurface(surface.name)}
              onPointerOut={() => setHoveredSurface(null)}
            />
            {createPVPanelMesh(surface)}
          </group>
        );
      })}
      
      {/* Render fenestration surfaces (windows, doors) */}
      {geometryData.fenestration_surfaces.map((surface, index) => {
        // Similar to ModelViewer implementation
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
        
        // Adjust positions
        const positions = geometry.attributes.position;
        for (let i = 0; i < positions.count; i++) {
          const x = positions.getX(i);
          const y = positions.getY(i);
          
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