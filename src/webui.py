import gradio as gr
from supabase import create_client, Client
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

class WebUI:
    def __init__(self):
        self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.table = self.client.table('IDF_DATA')
    
    def _get_filename(self):
        response = self.table.select('*').execute()
        return response.data
    
    def build_interface(self):
        data_dict = self._get_filename()[0]

        with gr.Blocks(title="IDF Data Viewer") as demo:

            global_state = gr.State(data_dict)
            gr.Markdown("## IDF Data Viewer ##")

            with gr.Row():
                # 左侧：用于选择并编辑 JSON 中的对象
                with gr.Column(scale=1):
                    gr.Markdown("### 左侧表单")

                    # 提取所有对象，并使用它们的 type 或者 index 作为下拉菜单选项
                    def get_object_choices(data):
                        """
                        根据 data 的内容动态生成下拉菜单选项:
                        形如: ["0. Version", "1. Timestep", ...]
                        """
                        objects = data.get("objects", [])
                        choices = []
                        for i, obj in enumerate(objects):
                            # 组合一下 type 和下标
                            t = obj.get("type", "NoType")
                            choices.append(f"{i}. {t}")
                        return choices
                    
                    obj_dropdown = gr.Dropdown(
                        label="选择要编辑的对象",
                        choices=get_object_choices(data_dict),
                        value=None,
                        interactive=True
                    )

                    # 针对对象的一些字段提供对应输入组件
                    obj_type = gr.Textbox(label="type")
                    obj_name = gr.Textbox(label="name")
                    obj_value = gr.Textbox(label="value")
                    obj_programline = gr.Textbox(
                        label="programline(用逗号或其它分隔符分开)",
                        placeholder="示例: Bldg,0.0,Suburbs"
                    )
                    obj_note = gr.Textbox(
                        label="note(用逗号或其它分隔符分开)",
                        placeholder="示例: Name,North Axis,Terrain..."
                    )
                    obj_units = gr.Textbox(
                        label="units(用逗号或其它分隔符分开)",
                        placeholder="示例: ,deg,,,deltaC"
                    )

                    def on_update_object(dropdown_val, data, t, nm, val, pline, note, units):
                        """
                        根据用户填写的组件值更新 global_state 中的对应对象。
                        """
                        if not dropdown_val:
                            return data, "没有可更新的对象"

                        objects = data.get("objects", [])
                        index_str = dropdown_val.split(".")[0]
                        try:
                            index = int(index_str)
                        except:
                            return data, "解析索引错误，无法更新。"

                        if index < 0 or index >= len(objects):
                            return data, f"索引 {index} 超出范围"

                        # 更新
                        objects[index]["type"] = t
                        objects[index]["name"] = nm
                        objects[index]["value"] = val

                        # 将用户输入的字符串再拆分为列表
                        objects[index]["programline"] = [x.strip() for x in pline.split(",")] if pline else []
                        objects[index]["note"] = [x.strip() for x in note.split(",")] if note else []
                        objects[index]["units"] = [x.strip() for x in units.split(",")] if units else []

                        msg = f"对象 {index} 已更新: type={t}, name={nm}"
                        return data, msg

                    update_btn = gr.Button("更新对象")

                    update_output = gr.Textbox(label="更新结果", interactive=False)

                    update_btn.click(
                        fn=on_update_object,
                        inputs=[obj_dropdown, global_state, obj_type, obj_name, obj_value, obj_programline, obj_note, obj_units],
                        outputs=[global_state, update_output]
                    )

                    # 提供一个按钮将修改写回文件
                    def on_save_file(data):
                        return "所有更改已保存到 1Zone.json！"

                    save_btn = gr.Button("保存至文件")
                    save_output = gr.Textbox(label="保存结果", interactive=False)

                    save_btn.click(
                        fn=on_save_file,
                        inputs=[global_state],
                        outputs=save_output
                    )

                # 右侧：比如可以做一个简单的“预览”或者“终端输出”区域
                with gr.Column(scale=2):
                    gr.Markdown("### 右侧预览/输出")
                    preview_box = gr.JSON(value=data_dict, label="JSON 文件预览")

                    def on_refresh_preview(data):
                        return data

                    refresh_button = gr.Button("刷新预览")
                    refresh_button.click(
                        fn=on_refresh_preview,
                        inputs=[global_state],
                        outputs=preview_box
                    )

        return demo
    
if __name__ == "__main__":
    webui = WebUI()
    app = webui.build_interface()
    app.launch()
