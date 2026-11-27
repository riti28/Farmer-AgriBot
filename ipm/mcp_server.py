"""
BONUS: MCP server so an AI assistant (Claude Desktop, MCP Inspector, ...) can use our IPM tools.

Tools:
  pest_risk(...)               -> neural-network pest risk for one crop and week of weather
  search_ipm_knowledge(query)  -> top-3 corpus passages with citations

Run:  python mcp_server.py          (starts the server over stdio)
Test: python mcp_server.py --test   (calls both tools once and prints the result)
"""
import sys
from mcp.server.fastmcp import FastMCP
import rag
from predict import predict_risk

mcp = FastMCP("farmer-agribot-ipm")
DISCLAIMER = "Not an agronomist. No yield guarantee. Consult your local agricultural extension officer."


@mcp.tool()
def pest_risk(crop: str, crop_stage: str, temp: float, humidity: float, rainfall: float) -> dict:
    """Predict this week's pest risk (Low / Medium / High) for a crop, anywhere in the world.

    Args:
        crop: one of rice, maize, cotton, sugarcane, finger_millet
        crop_stage: one of seedling, vegetative, flowering, maturity
        temp: mean air temperature of the week in deg C
        humidity: mean relative humidity of the week in %
        rainfall: total rainfall of the week in mm
    Returns:
        {"risk": label, "confidence": 0-1, "probabilities": {...}, "disclaimer": ...}
    """
    risk, confidence, probs = predict_risk(crop, crop_stage, temp, humidity, rainfall)
    return {"risk": risk, "confidence": round(confidence, 3), "probabilities": probs,
            "disclaimer": DISCLAIMER}


@mcp.tool()
def search_ipm_knowledge(query: str) -> dict:
    """Search our 49 IPM notes (FAO, ICAR, IRRI, CIMMYT, CABI, universities) with hybrid
    search (meaning embeddings + TF-IDF words), understanding everyday farmer words.

    Args:
        query: a pest question in plain English, e.g. "worms eating my maize"
    Returns:
        {"answer": 2-4 best sentences from the top-3 passages, cited [1][2][3]
                   (or "Not in my documents."),
         "citations": [{"doc_id", "title", "org", "url", "score"}, ...],
         "suggestions": closest topics we cover when nothing matched}
    """
    result = rag.answer(query)
    citations = [{k: s[k] for k in ("doc_id", "title", "org", "url", "score")} for s in result["sources"]]
    return {"answer": result["text"], "citations": citations, "suggestions": result["suggestions"]}


if __name__ == "__main__":
    if "--test" in sys.argv:
        print(pest_risk("rice", "vegetative", 26.0, 82.0, 25.0))
        print(search_ipm_knowledge("worms eating my maize"))
    else:
        mcp.run()  # stdio transport
