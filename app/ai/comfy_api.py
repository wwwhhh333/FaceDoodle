import json
import uuid
import urllib.request

def generate_transparent_sticker(prompt_text, server_address="127.0.0.1:8188"):
    """
    根据 transparent_workflow_api.json 的结构生成透明贴纸
    """
    # 1. 加载 API 格式的 JSON 工作流
    try:
        with open("../workflows/transparent_workflow_api.json", "r", encoding="utf-8") as f:
            workflow = json.load(f)
    except FileNotFoundError:
        print("Error: 找不到工作流 JSON 文件，请检查路径。")
        return None

    # 2. 动态修改正向提示词
    # 根据你的 JSON，ID "8" 是 CLIPTextEncode (Prompt)
    if "8" in workflow:
        # 保留用户核心意图，同时加入触发透明贴纸风格的关键提示词
        enhanced_prompt = f"{prompt_text}, sticker, cut_sticker, white background, high quality"
        workflow["8"]["inputs"]["text"] = enhanced_prompt
        print(f"核心提示词已更新: {enhanced_prompt}")
    else:
        print("Warning: 未在 JSON 中找到 ID 为 '8' 的提示词节点")

    # 3. 设置随机种子 (Seed)
    # 根据你的 JSON，ID "11" 是 KSampler
    seed = uuid.uuid4().int >> 64
    if "11" in workflow:
        workflow["11"]["inputs"]["seed"] = seed
        print(f"随机种子已设置: {seed}")
    else:
        print("Warning: 未在 JSON 中找到 ID 为 '11' 的采样器节点")

    # 4. 封装请求体
    # client_id 用于标识当前客户端，方便后续获取历史记录
    client_id = str(uuid.uuid4())
    p = {"prompt": workflow, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')

    # 5. 发送请求到 ComfyUI API 端口
    url = f"http://{server_address}/prompt"
    req = urllib.request.Request(url, data=data)

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            prompt_id = result.get("prompt_id")
            print(f"🚀 任务已提交至 ComfyUI! Prompt ID: {prompt_id}")
            return prompt_id
    except Exception as e:
        print(f"❌ 提交任务失败: {e}")
        return None

# 测试代码
if __name__ == "__main__":
    # 测试生成一个猫耳贴纸
    generate_transparent_sticker("one pair of cat ear, pink and white")