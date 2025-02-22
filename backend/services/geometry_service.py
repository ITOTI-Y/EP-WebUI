# backend/services/geometry_service.py

from fastapi import HTTPException

from .idf_service import get_idf_object, save_idf_file, MockUploadFile

def extract_vertices(surface):
    """
    辅助函数：提取给定 surface 的所有顶点坐标
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
    # 直接获取 idf_obj, 如有 HTTPException 则会自动上抛
    idf_obj = await get_idf_object(idf_id)

    # 使用列表推导式构造 zones 列表
    zones = [{"name": zone.Name} for zone in idf_obj.idfobjects["ZONE"]]

    # 使用列表推导式构造 surfaces 列表，并借助辅助函数获取顶点
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


if __name__ == "__main__":
    import asyncio
    import pathlib
    async def main_test_geometry():
        # 简化测试流程
        test_idf_path = pathlib.Path(__file__).parent.parent.parent / "data" / "test.idf"
        if not test_idf_path.exists():
            raise FileNotFoundError(f"Test IDF file not found at {test_idf_path}")

        with open(test_idf_path, "rb") as f:
            test_content = f.read()
        mock_file = MockUploadFile("test.idf", test_content)
        test_idf_id = await save_idf_file(mock_file)

        try:
            geometry_data = await get_geometry_data(test_idf_id)
            print(geometry_data)  # 输出结果便于查看
        except HTTPException as e:
            print(f"HTTPException: {e}")
    
    asyncio.run(main_test_geometry())
