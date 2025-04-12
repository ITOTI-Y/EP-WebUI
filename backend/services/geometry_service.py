# backend/services/geometry_service.py

from fastapi import HTTPException

from .idf_service import get_idf_object, save_idf_file, MockUploadFile

def extract_vertices(surface):
    """Extracts vertex coordinates for a given surface object.

    Args:
        surface: An object representing a surface (e.g., BUILDING SURFACE:DETAILED).

    Returns:
        A list of lists, where each inner list contains the [x, y, z] coordinates
        of a vertex.
    """
    num_vertices = int(surface.Number_of_Vertices)
    return [
        [
            float(getattr(surface, f"Vertex_{i}_Xcoordinate")),
            float(getattr(surface, f"Vertex_{i}_Ycoordinate")),
            float(getattr(surface, f"Vertex_{i}_Zcoordinate"))
        ]
        for i in range(1, num_vertices + 1)
    ]

async def get_geometry_data(idf_id: str):
    """Retrieves geometric data from a specified IDF object.

    Args:
        idf_id: The unique identifier for the IDF object.

    Returns:
        A dictionary containing lists of zones, building surfaces, and
        fenestration surfaces with their properties and vertices.

    Raises:
        HTTPException: If the IDF object cannot be retrieved.
    """
    # Retrieve IDF object; HTTPException is automatically propagated.
    idf_obj = await get_idf_object(idf_id)

    # Build zones list.
    zones = [{"name": zone.Name} for zone in idf_obj.idfobjects["ZONE"]]

    # Build surfaces list, extracting vertices using the helper function.
    surfaces = [
        {
            "name": surface.Name, # string
            "surface_type": surface.Surface_Type, # string
            "construction_name": surface.Construction_Name, # string
            "zone_name": surface.Zone_Name, # string
            "vertices": extract_vertices(surface) # list[list[float]]
        }
        for surface in idf_obj.idfobjects["BUILDINGSURFACE:DETAILED"]
    ]

    # Build fenestration surfaces list.
    fenestration_surfaces = [
        {
            "name": surface.Name, # string
            "surface_type": surface.Surface_Type, # string
            "vertices": extract_vertices(surface), # list[list[float]]
        }
        for surface in idf_obj.idfobjects["FENESTRATIONSURFACE:DETAILED"]
    ]

    geometry_data = {"zones": zones, "surfaces": surfaces, "fenestration_surfaces": fenestration_surfaces}

    return geometry_data

# Main execution block for testing.
if __name__ == "__main__":
    import asyncio
    import pathlib
    async def main_test_geometry():
        # Simplified test setup.
        test_idf_path = pathlib.Path(__file__).parent.parent.parent / "data" / "test.idf"
        if not test_idf_path.exists():
            raise FileNotFoundError(f"Test IDF file not found at {test_idf_path}")

        with open(test_idf_path, "rb") as f:
            test_content = f.read()
        mock_file = MockUploadFile("test.idf", test_content)
        test_idf_id = await save_idf_file(mock_file)

        try:
            geometry_data = await get_geometry_data(test_idf_id)
            # Print results for verification.
            print(geometry_data)
        except HTTPException as e:
            print(f"HTTPException: {e}")

    asyncio.run(main_test_geometry())
