from __future__ import annotations

import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.config import settings
from app.services.llm_followups import build_llm_style_followups

router = APIRouter()


@router.post("/chat_question")
async def chat_question(question: str = Form(...)):
    from rag_pipeline import get_answer

    answer, _ = get_answer(question)

    chips, suggest_live_chat, fu_source = build_llm_style_followups(
        user_question=question,
        answer_text=answer or "",
        k=4,
    )

    if suggest_live_chat:
        chips = [
            {"label": "Talk to an admin", "payload": {"type": "action", "action": "escalate"}}
        ]

    resp: Dict[str, Any] = {
        "answer": answer,
        "suggest_live_chat": suggest_live_chat,
        "suggested_followups": chips,
    }
    if settings.debug_followups:
        resp["followup_generator"] = fu_source
    return resp


@router.post("/chat_question_stream")
async def chat_question_stream(question: str = Form(...)):
    from rag_pipeline import get_answer_stream

    async def event_generator():
        full_answer = ""
        for chunk in get_answer_stream(question):
            full_answer += chunk
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

        chips, suggest_live_chat, fu_source = build_llm_style_followups(
            user_question=question,
            answer_text=full_answer or "",
            k=4,
        )

        if suggest_live_chat:
            chips = [
                {"label": "Talk to an admin", "payload": {"type": "action", "action": "escalate"}}
            ]

        followup_data: Dict[str, Any] = {
            "type": "followups",
            "suggest_live_chat": suggest_live_chat,
            "suggested_followups": chips,
        }

        if settings.debug_followups:
            followup_data["followup_generator"] = fu_source

        yield f"data: {json.dumps(followup_data)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class TicketAnalysisRequest(BaseModel):
    message: str


@router.post("/api/analyze_ticket")
async def analyze_ticket_request(request: TicketAnalysisRequest):
    try:
        from rag_pipeline import get_answer
        import re

        analysis_prompt = f"""
        Analyze the following user message and extract ticket information.

        User message: "{request.message}"

        Extract the following:
        1. Subject: A brief subject line (max 100 chars)
        2. Category: One of (Technical Support, Academic, Financial, Housing, Registration, Other)
        3. Priority: One of (Low, Medium, High) - based on urgency in the message
        4. A clear description of the issue

        Respond in this exact format:
        SUBJECT: [subject]
        CATEGORY: [category]
        PRIORITY: [priority]
        DESCRIPTION: [description]
        """

        answer, _ = get_answer(analysis_prompt)

        subject_match = re.search(r"SUBJECT:\s*(.+)", answer)
        category_match = re.search(r"CATEGORY:\s*(.+)", answer)
        priority_match = re.search(r"PRIORITY:\s*(.+)", answer)
        description_match = re.search(r"DESCRIPTION:\s*(.+)", answer, re.DOTALL)

        subject = subject_match.group(1).strip() if subject_match else "Support Request"
        category = category_match.group(1).strip() if category_match else "Other"
        priority = priority_match.group(1).strip() if priority_match else "Medium"
        description = description_match.group(1).strip() if description_match else request.message

        valid_categories = [
            "Technical Support",
            "Academic",
            "Financial",
            "Housing",
            "Registration",
            "Other",
        ]
        if category not in valid_categories:
            category = "Other"

        valid_priorities = ["Low", "Medium", "High"]
        if priority not in valid_priorities:
            priority = "Medium"

        return {
            "subject": subject[:100],
            "category": category,
            "priority": priority,
            "description": description,
        }
    except Exception as exc:
        print(f"Error analyzing ticket: {exc}")
        return {
            "subject": "Support Request",
            "category": "Other",
            "priority": "Medium",
            "description": request.message,
        }


class MapAnalysisRequest(BaseModel):
    message: str


@router.post("/api/analyze_map_request")
async def analyze_map_request(request: MapAnalysisRequest):
    try:
        message = request.message.lower()

        buildings = {
            "library": {
                "name": "Mary and Jeff Bell Library",
                "lat": 27.713788736691168,
                "lng": -97.32474868648656,
                "address": "Mary and Jeff Bell Library, TAMUCC",
                "description": "Main library with study spaces and research resources",
                "hours": "Mon-Fri 7:30am-11pm",
            },
            "university center": {
                "name": "University Center (UC)",
                "lat": 27.712071037382053,
                "lng": -97.3257065414334,
                "address": "University Center, TAMUCC",
                "description": "Student hub with dining, bookstore, and meeting spaces",
                "hours": "Mon-Fri 7am-10pm",
            },
            "uc": {
                "name": "University Center (UC)",
                "lat": 27.712071037382053,
                "lng": -97.3257065414334,
                "address": "University Center, TAMUCC",
                "description": "Student hub with dining, bookstore, and meeting spaces",
                "hours": "Mon-Fri 7am-10pm",
            },
            "dining": {
                "name": "Islander Dining",
                "lat": 27.711621676963894,
                "lng": -97.32258737277509,
                "address": "Islander Dining, TAMUCC",
                "description": "Main dining hall with multiple food stations",
                "hours": "Daily 7am-9pm",
            },
            "islander dining": {
                "name": "Islander Dining",
                "lat": 27.711621676963894,
                "lng": -97.32258737277509,
                "address": "Islander Dining, TAMUCC",
                "description": "Main dining hall with multiple food stations",
                "hours": "Daily 7am-9pm",
            },
            "natural resources": {
                "name": "Natural Resources Center (NRC)",
                "lat": 27.715332468715157,
                "lng": -97.32880933649331,
                "address": "Natural Resources Center, TAMUCC",
                "description": "Environmental science and research facility",
                "hours": "Mon-Fri 8am-5pm",
            },
            "nrc": {
                "name": "Natural Resources Center (NRC)",
                "lat": 27.715332468715157,
                "lng": -97.32880933649331,
                "address": "Natural Resources Center, TAMUCC",
                "description": "Environmental science and research facility",
                "hours": "Mon-Fri 8am-5pm",
            },
            "engineering": {
                "name": "Engineering Building",
                "lat": 27.712772225261283,
                "lng": -97.32565431063824,
                "address": "Engineering Building, TAMUCC",
                "description": "College of Engineering classrooms and labs",
                "hours": "Mon-Fri 8am-6pm",
            },
            "corpus christi hall": {
                "name": "Corpus Christi Hall (CCH)",
                "lat": 27.71516058584113,
                "lng": -97.32370567166191,
                "address": "Corpus Christi Hall, TAMUCC",
                "description": "Admissions, financial aid, and student services",
                "hours": "Mon-Fri 8am-5pm",
            },
            "cch": {
                "name": "Corpus Christi Hall (CCH)",
                "lat": 27.71516058584113,
                "lng": -97.32370567166191,
                "address": "Corpus Christi Hall, TAMUCC",
                "description": "Admissions, financial aid, and student services",
                "hours": "Mon-Fri 8am-5pm",
            },
            "student services": {
                "name": "Student Services Center",
                "lat": 27.71374042156452,
                "lng": -97.32390201020142,
                "address": "Student Services Center, TAMUCC",
                "description": "Student support services and administration",
                "hours": "Mon-Fri 8am-5pm",
            },
            "bay hall": {
                "name": "Bay Hall",
                "lat": 27.713613491472024,
                "lng": -97.32348514338884,
                "address": "Bay Hall, TAMUCC",
                "description": "Business college classrooms and faculty offices",
                "hours": "Mon-Fri 8am-5pm",
            },
            "sciences": {
                "name": "Center for the Sciences",
                "lat": 27.712809298665885,
                "lng": -97.32486990268086,
                "address": "Center for the Sciences, TAMUCC",
                "description": "Science labs and classrooms",
                "hours": "Mon-Fri 8am-6pm",
            },
            "center for sciences": {
                "name": "Center for the Sciences",
                "lat": 27.712809298665885,
                "lng": -97.32486990268086,
                "address": "Center for the Sciences, TAMUCC",
                "description": "Science labs and classrooms",
                "hours": "Mon-Fri 8am-6pm",
            },
            "education": {
                "name": "College of Education and Human Development",
                "lat": 27.713186318706956,
                "lng": -97.32428916719182,
                "address": "College of Education and Human Development, TAMUCC",
                "description": "Education college offices and classrooms",
                "hours": "Mon-Fri 8am-5pm",
            },
            "faculty center": {
                "name": "Faculty Center",
                "lat": 27.712820723536026,
                "lng": -97.32358260567656,
                "address": "Faculty Center, TAMUCC",
                "description": "Faculty offices and meeting rooms",
                "hours": "Mon-Fri 8am-5pm",
            },
            "wellness": {
                "name": "Dugan Wellness Center",
                "lat": 27.711601112024837,
                "lng": -97.32413753070178,
                "address": "Dugan Wellness Center, TAMUCC",
                "description": "Student health services and counseling",
                "hours": "Mon-Fri 8am-5pm",
            },
            "dugan": {
                "name": "Dugan Wellness Center",
                "lat": 27.711601112024837,
                "lng": -97.32413753070178,
                "address": "Dugan Wellness Center, TAMUCC",
                "description": "Student health services and counseling",
                "hours": "Mon-Fri 8am-5pm",
            },
            "health": {
                "name": "Dugan Wellness Center",
                "lat": 27.711601112024837,
                "lng": -97.32413753070178,
                "address": "Dugan Wellness Center, TAMUCC",
                "description": "Student health services and counseling",
                "hours": "Mon-Fri 8am-5pm",
            },
            "business": {
                "name": "College of Business",
                "lat": 27.714591440638948,
                "lng": -97.32466461335527,
                "address": "College of Business, TAMUCC",
                "description": "College of Business and entrepreneurship programs",
                "hours": "Mon-Fri 8am-5pm",
            },
            "tidal hall": {
                "name": "Tidal Hall",
                "lat": 27.715529412703646,
                "lng": -97.32710819211944,
                "address": "Tidal Hall, TAMUCC",
                "description": "Student housing residence hall",
                "hours": "24/7 for residents",
            },
            "harte": {
                "name": "Harte Research Institute",
                "lat": 27.713459500631362,
                "lng": -97.32815759566772,
                "address": "Harte Research Institute, TAMUCC",
                "description": "Gulf of Mexico research and marine science",
                "hours": "Mon-Fri 8am-5pm",
            },
            "counseling": {
                "name": "University Counseling Center",
                "lat": 27.712490577148014,
                "lng": -97.32168122550681,
                "address": "University Counseling Center, TAMUCC",
                "description": "Mental health and counseling services for students",
                "hours": "Mon-Fri 8am-5pm",
            },
            "counseling center": {
                "name": "University Counseling Center",
                "lat": 27.712490577148014,
                "lng": -97.32168122550681,
                "address": "University Counseling Center, TAMUCC",
                "description": "Mental health and counseling services for students",
                "hours": "Mon-Fri 8am-5pm",
            },
        }

        for key, building in buildings.items():
            if key in message or building["name"].lower() in message:
                return {
                    "location": building,
                    "description": f"📍 Here's the location of the **{building['name']}**. {building['description']}.",
                }

        return {
            "location": None,
            "description": "Here's the TAMUCC campus map showing all major buildings.",
        }

    except Exception as exc:
        logging.error(f"Error analyzing map request: {exc}")
        raise HTTPException(status_code=500, detail="Failed to analyze map request")


class RoutingRequest(BaseModel):
    message: str


@router.post("/api/analyze_routing_request")
async def analyze_routing_request(request: RoutingRequest):
    try:
        message = request.message.lower()

        buildings = {
            "library": {"name": "Mary and Jeff Bell Library", "lat": 27.713788736691168, "lng": -97.32474868648656},
            "university center": {"name": "University Center (UC)", "lat": 27.712071037382053, "lng": -97.3257065414334},
            "uc": {"name": "University Center (UC)", "lat": 27.712071037382053, "lng": -97.3257065414334},
            "dining": {"name": "Islander Dining", "lat": 27.711621676963894, "lng": -97.32258737277509},
            "islander dining": {"name": "Islander Dining", "lat": 27.711621676963894, "lng": -97.32258737277509},
            "natural resources": {"name": "Natural Resources Center (NRC)", "lat": 27.715332468715157, "lng": -97.32880933649331},
            "nrc": {"name": "Natural Resources Center (NRC)", "lat": 27.715332468715157, "lng": -97.32880933649331},
            "engineering": {"name": "Engineering Building", "lat": 27.712772225261283, "lng": -97.32565431063824},
            "corpus christi hall": {"name": "Corpus Christi Hall (CCH)", "lat": 27.71516058584113, "lng": -97.32370567166191},
            "cch": {"name": "Corpus Christi Hall (CCH)", "lat": 27.71516058584113, "lng": -97.32370567166191},
            "student services": {"name": "Student Services Center", "lat": 27.71374042156452, "lng": -97.32390201020142},
            "bay hall": {"name": "Bay Hall", "lat": 27.713613491472024, "lng": -97.32348514338884},
            "sciences": {"name": "Center for the Sciences", "lat": 27.712809298665885, "lng": -97.32486990268086},
            "center for sciences": {"name": "Center for the Sciences", "lat": 27.712809298665885, "lng": -97.32486990268086},
            "education": {"name": "College of Education and Human Development", "lat": 27.713186318706956, "lng": -97.32428916719182},
            "faculty center": {"name": "Faculty Center", "lat": 27.712820723536026, "lng": -97.32358260567656},
            "wellness": {"name": "Dugan Wellness Center", "lat": 27.711601112024837, "lng": -97.32413753070178},
            "dugan": {"name": "Dugan Wellness Center", "lat": 27.711601112024837, "lng": -97.32413753070178},
            "health": {"name": "Dugan Wellness Center", "lat": 27.711601112024837, "lng": -97.32413753070178},
            "business": {"name": "College of Business", "lat": 27.714591440638948, "lng": -97.32466461335527},
            "tidal hall": {"name": "Tidal Hall", "lat": 27.715529412703646, "lng": -97.32710819211944},
            "harte": {"name": "Harte Research Institute", "lat": 27.713459500631362, "lng": -97.32815759566772},
            "counseling": {"name": "University Counseling Center", "lat": 27.712490577148014, "lng": -97.32168122550681},
            "counseling center": {"name": "University Counseling Center", "lat": 27.712490577148014, "lng": -97.32168122550681},
        }

        routing_patterns = [
            ("from", "to"),
            ("between", "and"),
            ("get to", "from"),
        ]

        origin = None
        destination = None

        for pattern in routing_patterns:
            if pattern[0] in message and pattern[1] in message:
                parts = message.split(pattern[0])
                if len(parts) > 1:
                    second_part = parts[1].split(pattern[1])
                    if len(second_part) > 1:
                        origin_text = second_part[0].strip()
                        dest_text = second_part[1].strip()

                        for key, building in buildings.items():
                            if key in origin_text or key == origin_text:
                                origin = building
                            if key in dest_text or key == dest_text:
                                destination = building

        if origin and destination:
            return {
                "origin": origin,
                "destination": destination,
                "found": True,
            }
        return {
            "origin": None,
            "destination": None,
            "found": False,
            "message": "I couldn't identify both the origin and destination buildings. Please specify like 'directions from Library to UC' or 'how to get from NRC to Wellness Center'.",
        }

    except Exception as exc:
        logging.error(f"Error analyzing routing request: {exc}")
        raise HTTPException(status_code=500, detail="Failed to analyze routing request")
