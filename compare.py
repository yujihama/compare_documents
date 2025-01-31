import os
import math
import pandas as pd
from typing import List, Dict, Union
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
import logging  # ログ機能の追加
from datetime import datetime  # datetimeモジュールを追加

# .envファイルから環境変数を読み込む
load_dotenv()

# ログ設定
logging.basicConfig(
    filename='compare_tool.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logging.getLogger('dotenv').setLevel(logging.WARNING)  # dotenvのログレベルを下げる

# OpenAIクライアントの初期化 (APIキーは環境変数から自動で読み込まれる)
client = OpenAI()

# Embedding用モデルの指定
EMBEDDING_MODEL = "text-embedding-3-large"

# 差分要約に使うチャットモデル
CHAT_MODEL = "gpt-4o"

# Pydanticモデル：GPT-4oの応答を構造化する
class CompareOutput(BaseModel):
    is_similar: bool
    difference_summary: str

def get_embedding(text: str) -> List[float]:
    """
    テキストを Embedding ベクトルに変換する関数。
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    # OpenAI pythonパッケージの形式に合わせて取得
    embedding = response.data[0].embedding
    return embedding

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    コサイン類似度を計算する。
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def compare_texts_with_gpt4o(base_text: str, compare_texts: Union[str, List[str]]) -> CompareOutput:
    """
    GPT-4o を使って 本社規定テキスト と 子会社規定テキスト/テキスト群 を比較し、
    似ているかどうか (is_similar) と差分サマリ (difference_summary) を
    構造化データとして返す。
    子会社規定が複数のチャンクに分割されている場合も対応。
    """
    messages = [
        {
            "role": "system",
            "content": (
                "あなたは法務の専門家アシスタントです。"
                "出力はJSON形式で、is_similarとdifference_summaryの2項目を返してください。"
                "これから提示する【本社規定項目】と【子会社規定】を比較し、"
                "子会社規定が本社規定の内容を実質的に満たしているか判断してください。"
                "もし内容が実質的に同じ(本社規定を満たしている)と判断できる場合はis_similar=Trueにしてください。"
                "一部でも本社規定を満たしていない場合はis_similar=Falseとしてください。"
                "本社規定にない内容が子会社規定にあっても、無視してください。"
                "子会社規定は複数のチャンクに分割されている可能性があります。すべてのチャンクを考慮して判断してください。" # 複数チャンク対応指示
            )
        },
        {
            "role": "user",
            "content": (
                "以下の【本社規定項目】の内容を【子会社規定】が満たしているか比較してください。\n"
                "【子会社規定】は複数のテキストチャンクで構成されている場合があります。\n" # 複数チャンクの説明を追加
                "すべてのチャンクを考慮して、全体として本社規定を満たしているか判断してください。\n" # 判断基準を明確化
                "もし文面が違う場合は、その違いを差分サマリとして簡潔に教えてください。\n"
                "出力はJSON形式で日本語で返してください。\n\n"
                "【本社規定項目】\n"
                f"{base_text}\n\n"
                "【子会社規定】\n"
                f"{chr(10).join(compare_texts) if isinstance(compare_texts, list) else compare_texts}\n" # 複数チャンクを改行で連結
            )
        }
    ]

    completion = client.beta.chat.completions.parse(
        model=CHAT_MODEL,
        messages=messages,
        response_format=CompareOutput,
    )
    return completion.choices[0].message.parsed

def generate_markdown_report(
    checklist: List[Dict],
    subsidiary_chunks: List[Dict],
    top_k: int = 3,
    similarity_threshold: float = 0.5
) -> str:
    """
    チェックリスト (本社規定の項目一覧) と 子会社規定のチャンク を比較して、
    Embeddings で候補を絞りつつ GPT-4o で差分要約。
    結果を Markdown で返す。

    改善点:
    1. 閾値を超えたチャンクはすべて関連チャンクとみなし、GPT-4oに問い合わせる。
    2. 1件も閾値以上がなければ「本社規定のみ」として扱う。
    3. GPT-4o応答(is_similar)がTrueのものが1つでもあれば「該当あり」とみなす。
    """

    logging.info("レポート生成開始")

    # 子会社チャンクに embedding が無い場合はここで埋め込む
    for chunk in subsidiary_chunks:
        if "embedding" not in chunk:
            chunk["embedding"] = get_embedding(chunk["text"])

    md_report = []
    md_report.append("# チェックリスト比較レポート\n")

    for item in checklist:
        item_id = item["id"]
        item_text = item["text"]

        logging.info(f"チェック項目ID: {item_id} の比較開始")
        logging.debug(f"本社規定項目テキスト: {item_text}")

        # 本社規定アイテムの embedding
        item_embedding = get_embedding(item_text)

        # 子会社規定チャンクとの類似度を計算 → 全件スコアリング
        scored_chunks = []
        for chunk in subsidiary_chunks:
            sim = cosine_similarity(item_embedding, chunk["embedding"])
            scored_chunks.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "similarity": sim
            })

        # 類似度でソート
        scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)

        # (1) 閾値以上のチャンクだけをリストアップ
        related_chunks = [c for c in scored_chunks if c["similarity"] >= similarity_threshold]
        # 閾値を超えるものがなければ、「本社規定にのみ存在」として扱う
        if not related_chunks:
            md_report.append(f"## チェック項目: {item_id}")
            md_report.append(f"**本社規定項目テキスト**: {item_text}\n")
            md_report.append("- 対応チャンク: **該当なし**（本社規定にのみ存在）\n")
            md_report.append("---")
            logging.info(f"チェック項目ID: {item_id} は関連チャンクなしと判定。")
            continue

        # (2) 閾値以上のチャンクがある場合、GPT-4o で順次判定
        found_fulfilled = False   # 一つでも is_similar = True があったか
        chunk_results = []        # 全チャンクの結果を保持

        # 上位N件に限定したい場合はここで[:top_k]にする
        related_chunk_texts = [c["text"] for c in related_chunks] # 関連チャンクのテキストをリストで取得
        if related_chunk_texts: # 関連チャンクが1つ以上存在する場合のみGPT-4o比較
            logging.debug(f"GPT-4o比較開始: {related_chunks}")
            compare_result = compare_texts_with_gpt4o(item_text, related_chunk_texts) # 複数チャンクをまとめて比較
            chunk_results.append({
                "chunk_ids": [c["id"] for c in related_chunks], # チャンクIDをリストで保持
                "similarities": [c["similarity"] for c in related_chunks], # 類似度をリストで保持
                "is_similar": compare_result.is_similar,
                "difference_summary": compare_result.difference_summary
            })
            if compare_result.is_similar:
                found_fulfilled = True

        # Markdown出力
        md_report.append(f"## チェック項目: {item_id}")
        md_report.append(f"**本社規定項目テキスト**: {item_text}\n")

        if found_fulfilled:
            md_report.append("- **この項目は子会社規定で満たされています。**")
        else:
            md_report.append("- **この項目は子会社規定で不十分と判定されました。**")

        # 関連チャンクの比較結果一覧を表示
        md_report.append("#### 関連チャンクの判定結果 (閾値以上)")
        for cr in chunk_results:
            md_report.append(f"- チャンクID: {', '.join(cr['chunk_ids'])} (類似度: {', '.join([str(round(s, 3)) for s in cr['similarities']])})") # 複数チャンクIDと類似度を表示
            md_report.append(f"  - is_similar: {cr['is_similar']}")
            if cr["difference_summary"]:
                md_report.append(f"  - 差分: {cr['difference_summary']}")
            md_report.append("")

        md_report.append("---")

    logging.info("レポート生成完了")
    return "\n".join(md_report)

# Streamlit アプリケーション構築
st.title("規定比較ツール")

st.header("1. 本社規定チェックリスト")
checklist_text = st.text_area(
    "本社規定のチェックリスト項目を1行ずつ入力してください。",
    """この規程は、会社の目的および基本方針を定めるものである。
適用範囲は全従業員とし、正社員および契約社員を含む。""",
    height=200
)
example_checklist = [
    {"id": f"1-{i+1}", "text": item.strip()}
    for i, item in enumerate(checklist_text.split('\n')) if item.strip()
]

st.header("2. 子会社規定チャンク")
subsidiary_chunks_text = st.text_area(
    "子会社規定のテキストチャンクを1行ずつ入力してください。",
    """当社の目的および基本方針は、企業価値の最大化と従業員の幸福を実現することにある。
この規程は、取締役および正社員に適用される。派遣社員は対象外とする。""",
    height=200
)
example_subsidiary_chunks = [
    {"id": f"C-{i+1}", "text": item.strip()}
    for i, item in enumerate(subsidiary_chunks_text.split('\n')) if item.strip()
]

st.sidebar.header("パラメータ設定")
top_k = st.sidebar.number_input("Top K", min_value=1, max_value=10, value=3)
similarity_threshold = st.sidebar.slider("類似度閾値", 0.0, 1.0, 0.5, step=0.1)

if st.button("比較実行"):
    if not example_checklist or not example_subsidiary_chunks:
        st.error("チェックリストと子会社規定チャンクの両方を入力してください。")
    else:
        with st.spinner("比較レポート作成中..."):
            report_md = generate_markdown_report(
                checklist=example_checklist,
                subsidiary_chunks=example_subsidiary_chunks,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
        st.header("比較レポート")
        st.markdown(report_md)
        st.success("比較レポートが完成しました！")

        # resultフォルダを作成（存在しない場合）
        os.makedirs("result", exist_ok=True)

        # 現在日時をファイル名に含める
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"result/compare_report_{now}.md"

        # レポートをファイルに保存
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(report_md)
        st.info(f"レポートを {file_name} に保存しました。") # 保存完了メッセージ

if __name__ == "__main__":
    pass
