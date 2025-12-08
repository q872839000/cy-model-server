"""
API 测试脚本

使用方法:
1. 先启动服务: python main.py
2. 运行测试: python test_api.py

测试内容:
- 健康检查
- 模型列表
- Embedding 生成
- 聊天补全（非流式）
- 聊天补全（流式）
"""

import requests
import json
import sys

# 服务地址
BASE_URL = "http://localhost:8000"


def test_health():
    """测试健康检查接口"""
    print("\n" + "=" * 50)
    print("测试: 健康检查 GET /healthz")
    print("=" * 50)
    
    try:
        resp = requests.get(f"{BASE_URL}/healthz", timeout=5)
        print(f"状态码: {resp.status_code}")
        print(f"响应: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
        return resp.status_code == 200
    except Exception as e:
        print(f"错误: {e}")
        return False


def test_list_models():
    """测试模型列表接口"""
    print("\n" + "=" * 50)
    print("测试: 模型列表 GET /v1/models")
    print("=" * 50)
    
    try:
        resp = requests.get(f"{BASE_URL}/v1/models", timeout=5)
        print(f"状态码: {resp.status_code}")
        print(f"响应: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
        return resp.status_code == 200
    except Exception as e:
        print(f"错误: {e}")
        return False


def test_embeddings(model_name: str = None):
    """测试 Embedding 生成接口"""
    print("\n" + "=" * 50)
    print("测试: Embedding 生成 POST /v1/embeddings")
    print("=" * 50)
    
    # 如果没有指定模型，先获取可用模型
    if not model_name:
        try:
            resp = requests.get(f"{BASE_URL}/v1/models", timeout=5)
            models = resp.json().get("data", [])
            # 查找 embedding 模型（通常包含 bge、embed 等关键词）
            for m in models:
                mid = m.get("id", "").lower()
                if "bge" in mid or "embed" in mid:
                    model_name = m.get("id")
                    break
            if not model_name and models:
                model_name = models[0].get("id")
        except:
            pass
    
    if not model_name:
        print("跳过: 没有可用的 Embedding 模型")
        return True
    
    payload = {
        "model": model_name,
        "input": ["你好，世界！", "Hello, World!"]
    }
    
    try:
        print(f"请求模型: {model_name}")
        print(f"请求体: {json.dumps(payload, indent=2, ensure_ascii=False)}")
        
        resp = requests.post(
            f"{BASE_URL}/v1/embeddings",
            json=payload,
            timeout=30
        )
        print(f"状态码: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            # 只显示向量维度，不显示完整向量
            for item in data.get("data", []):
                dim = len(item.get("embedding", []))
                print(f"  文本 {item.get('index')}: 向量维度 = {dim}")
            return True
        else:
            print(f"响应: {resp.text}")
            return False
    except Exception as e:
        print(f"错误: {e}")
        return False


def test_chat_completion(model_name: str = None, stream: bool = False):
    """测试聊天补全接口"""
    mode = "流式" if stream else "非流式"
    print("\n" + "=" * 50)
    print(f"测试: 聊天补全({mode}) POST /v1/chat/completions")
    print("=" * 50)
    
    # 如果没有指定模型，先获取可用模型
    if not model_name:
        try:
            resp = requests.get(f"{BASE_URL}/v1/models", timeout=5)
            models = resp.json().get("data", [])
            # 查找 LLM 模型（排除 bge、embed、rerank 等）
            for m in models:
                mid = m.get("id", "").lower()
                if not any(kw in mid for kw in ["bge", "embed", "rerank"]):
                    model_name = m.get("id")
                    break
        except:
            pass
    
    if not model_name:
        print("跳过: 没有可用的 LLM 模型")
        return True
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "你是一个有帮助的助手。"},
            {"role": "user", "content": "请用一句话介绍自己。"}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": stream
    }
    
    try:
        print(f"请求模型: {model_name}")
        print(f"流式模式: {stream}")
        
        if stream:
            # 流式请求
            resp = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=60
            )
            print(f"状态码: {resp.status_code}")
            print("响应内容:")
            
            full_content = ""
            for line in resp.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            print("\n[DONE]")
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                print(content, end="", flush=True)
                                full_content += content
                        except json.JSONDecodeError:
                            pass
            
            print(f"\n\n完整响应: {full_content}")
            return resp.status_code == 200
        else:
            # 非流式请求
            resp = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            print(f"状态码: {resp.status_code}")
            
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"响应内容: {content}")
                return True
            else:
                print(f"响应: {resp.text}")
                return False
    except Exception as e:
        print(f"错误: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 50)
    print("CY Model Server API 测试")
    print("=" * 50)
    print(f"服务地址: {BASE_URL}")
    
    results = {}
    
    # 1. 健康检查
    results["健康检查"] = test_health()
    if not results["健康检查"]:
        print("\n服务未启动，请先运行: python main.py")
        sys.exit(1)
    
    # 2. 模型列表
    results["模型列表"] = test_list_models()
    
    # 3. Embedding 生成
    results["Embedding生成"] = test_embeddings()
    
    # 4. 聊天补全（非流式）
    results["聊天补全(非流式)"] = test_chat_completion(stream=False)
    
    # 5. 聊天补全（流式）
    results["聊天补全(流式)"] = test_chat_completion(stream=True)
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("所有测试通过!" if all_passed else "部分测试失败"))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
