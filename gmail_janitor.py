import os
import time
import json
import base64
from bs4 import BeautifulSoup

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from openai import OpenAI

# ==============================================================================
# CONFIGURATION & GLOBAL RULES
# ==============================================================================

# 🚨 DRY RUN TOGGLE 🚨
DRY_RUN = True 

# Time-Based Rules
SAFETY_DAYS = 30       
MAX_INBOX_DAYS = 60    
MAX_IMPORTANT_DAYS = 180

# State Tracking Label
PROCESSED_LABEL = "AI_Processed"

# Strict Label Whitelist (Removed 'Important', added 'Correspondence' & 'Official')
VALID_LABELS = {
    "Correspondence", "Official", "Social", "Work", "Purchases", 
    "Bills", "Accounts", "Statements", "Marketing", "Newsletters", "Junk"
}

# Overrides: Labels to archive immediately, UNLESS the LLM flagged them as Important
IMMEDIATE_ARCHIVE_LABELS = {"Marketing", "Junk", "Newsletters"}

# Llama-Server / LLM Configuration
LLM_BASE_URL = "http://localhost:8080/v1"
LLM_API_KEY = "sk-local-no-key-needed"
LLM_MODEL = "local-model" 

LABEL_CACHE = {}
llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

# ==============================================================================
# GMAIL API UTILITIES
# ==============================================================================
# (Authentication and parsing functions remain unchanged)

SCOPES =['https://www.googleapis.com/auth/gmail.modify']

def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def get_email_body(payload):
    body_text = ""
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain':
                data = part['body'].get('data')
                if data: return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            elif part['mimeType'] == 'text/html':
                data = part['body'].get('data')
                if data:
                    html = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                    return BeautifulSoup(html, 'html.parser').get_text(separator=' ', strip=True)
            elif 'parts' in part:
                return get_email_body(part)
    else:
        data = payload['body'].get('data')
        if data:
            decoded = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            if payload['mimeType'] == 'text/html':
                 return BeautifulSoup(decoded, 'html.parser').get_text(separator=' ', strip=True)
            return decoded
    return body_text

def get_or_create_label(service, label_name):
    if not label_name: return None
    if label_name in LABEL_CACHE: return LABEL_CACHE[label_name]

    try:
        results = service.users().labels().list(userId='me').execute()
        labels = results.get('labels', [])
        
        for label in labels:
            if label['name'].lower() == label_name.lower():
                LABEL_CACHE[label_name] = label['id']
                return label['id']
        
        if DRY_RUN:
            print(f"  [DRY RUN] Would CREATE new Gmail label: '{label_name}'")
            LABEL_CACHE[label_name] = f"DRY_RUN_MOCK_ID_{label_name}"
            return LABEL_CACHE[label_name]
        else:
            print(f"  Creating missing label: {label_name}")
            new_label = service.users().labels().create(
                userId='me', 
                body={'name': label_name, 'labelListVisibility': 'labelShow', 'messageListVisibility': 'show'}
            ).execute()
            LABEL_CACHE[label_name] = new_label['id']
            return new_label['id']
            
    except Exception as e:
        print(f"  Error handling label '{label_name}': {e}")
        return None

# ==============================================================================
# LLM LOGIC & PROMPT PROGRAM
# ==============================================================================
#   "reasoning": "1 sentence explanation.",


def evaluate_email_with_llm(sender, subject, body):
    system_prompt = f"""You are an intelligent email classifier. Read the email and output ONLY a valid JSON object. Do not include markdown formatting.

VALID LABELS:
- Correspondence: Personal emails from friends, family, or real humans. Direct replies to things I've sent.
- Official: Government notices, legal, housing, or crucial administrative matters.
- Social: Social media notifications.
- Work: Professional communications.
- Purchases: Receipts, order confirmations, shipping updates.
- Bills: Due bills, invoices, automated bill payments.
- Accounts: Non-promotional information, security alerts, or customer information from an entity I have an account with.
- Statements: Bank/credit card monthly statements
- Newsletters: Substack, TLDR, Prestige Journalism
- Marketing: Ads, Promotions
- Junk: Spam, political campaigns, cold sales outreach.

OUTPUT SCHEMA:
{{
  "label_name": "Exact Label Name from the list above",
  "important": true or false
}}

INSTRUCTIONS FOR "important" FLAG:
Mark "important": true ONLY if the recipient absolutely must see this. Examples include due debts, threatened legal action, heartfelt letters from loved ones, job offers (not just a job posting), or unclear but highly time-sensitive personal matters. If it's just a routine receipt or a newsletter, set it to false. When in absolute doubt about whether I need to see it, default to true.
"""
    user_prompt = f"SENDER: {sender}\nSUBJECT: {subject}\nBODY PREVIEW: {body[:1500]}"

    try:
        response = llm_client.chat.completions.with_raw_response.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            extra_body={
                "thinking_budget_tokens": 0        # ← this is the "no reasoning" flag
            },
            # chat_template_kwargs= {
            #     "enable_thinking": False
            # },
            temperature=0.0
        )
        response_data = json.loads(response.text)

        # Extract Standard Usage
        usage = response_data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        # Extract custom llama.cpp hardware timings
        timings = response_data.get("timings", {})
        
        if timings:
            time_reading = timings.get("prompt_ms", 0) / 1000.0
            time_writing = timings.get("predicted_ms", 0) / 1000.0
            
            # prompt_n is what llama.cpp ACTUALLY evaluated (ignoring cache)
            evaluated_tokens = timings.get("prompt_n", input_tokens)
            cached_tokens = input_tokens - evaluated_tokens
            
            tps_reading = timings.get("prompt_per_second", evaluated_tokens / time_reading if time_reading else 0)
            tps_writing = timings.get("predicted_per_second", output_tokens / time_writing if time_writing else 0)
            
            # Calculate "Effective" TPS to show how fast it feels from a user perspective
            effective_tps_reading = input_tokens / time_reading if time_reading else 0
            
            print("\n" + "="*50)
            print("📊 llama.cpp Server-Side Metrics")
            print("="*50)
            print(f"Total Input Tokens:   {input_tokens} ({cached_tokens} cached, {evaluated_tokens} evaluated)")
            print(f"Total Output Tokens:  {output_tokens}")
            print(f"Time Reading Input:   {time_reading:.3f} s")
            print(f"Time Writing Output:  {time_writing:.3f} s")
            print(f"TPS (Reading):        {tps_reading:.2f} t/s (Hardware actual)")
            if cached_tokens > 0:
                print(f"TPS (Effective Read): {effective_tps_reading:.2f} t/s (Due to Cache)")
            print(f"TPS (Writing):        {tps_writing:.2f} t/s")
            print("="*50 + "\n")

        raw_response = response_data["choices"][0]["message"]["content"].strip()
        
        # Clean markdown
        if raw_response.startswith('```json'): raw_response = raw_response[7:]
        if raw_response.startswith('```'): raw_response = raw_response[3:]
        if raw_response.endswith('```'): raw_response = raw_response[:-3]
        
        decision = json.loads(raw_response.strip())

        # 1. Enforce strict boolean for 'important'
        val = decision.get('important', True)
        if isinstance(val, str):
            decision['important'] = val.strip().lower() == 'true'
        elif not isinstance(val, bool):
            decision['important'] = True
            
        # 2. Enforce label whitelist
        extracted_label = decision.get('label_name')
        if extracted_label not in VALID_LABELS:
            print(f"  [!] LLM Hallucinated invalid label: '{extracted_label}'. Defaulting to 'Correspondence'.")
            decision['label_name'] = 'Correspondence'
            decision['important'] = True  
            decision['reasoning'] = "Fallback triggered due to LLM label hallucination."

        return decision
        
    except json.JSONDecodeError as e:
        print(f"  [!] LLM JSON Error. Exception: {e}")
        return None
    except Exception as e:
        print(f"  [!] LLM Error: {e}")
        return None

# ==============================================================================
# BATCH PROCESSOR ENGINE
# ==============================================================================

def process_batch(service, query, batch_size, process_function):
    print(f"\nSearching: '{query}'")
    try:
        results = service.users().messages().list(userId='me', q=query, maxResults=batch_size).execute()
        messages = results.get('messages',[])
        
        if not messages:
            print("  No emails found matching query.")
            return False

        print(f"  Processing batch of {len(messages)} messages...\n")

        for msg in messages:
            try:
                msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
                headers = msg_data['payload']['headers']
                subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
                sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown Sender')
                
                process_function(service, msg['id'], sender, subject, msg_data)

            except HttpError as error:
                if error.resp.status in[403, 429]:
                    print("  API Rate limit hit. Backing off 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"  Unexpected API error: {error}")
            except Exception as e:
                print(f"  Failed to process message {msg['id']}: {e}")
                continue 
                
        return True
    except Exception as e:
        print(f"  Failed to fetch batch: {e}")
        return False

# ==============================================================================
# JOB LOGIC: What happens to the emails
# ==============================================================================

def logic_archive_only(service, msg_id, sender, subject, msg_data):
    """Used for time-based sweeping. Removes from INBOX blindly."""
    if DRY_RUN:
        print(f"  [DRY RUN] Would REMOVE from INBOX: {subject[:50]}")
        return
        
    service.users().messages().modify(
        userId='me', id=msg_id, body={'removeLabelIds': ['INBOX']}
    ).execute()
    print(f"  [Archived] {subject[:50]}")

def logic_llm_triage(service, msg_id, sender, subject, msg_data):
    """The main AI sorter. Applies category and state labels."""
    body = get_email_body(msg_data['payload'])
    
    print(f"Evaluating: {subject[:50]}\n      From: {sender[:50]}")
    decision = evaluate_email_with_llm(sender, subject, body)
    
    if not decision:
        print("  └─> [Skipped due to LLM error]\n")
        return

    label_name = decision.get('label_name')
    is_important = decision.get('important')
    
    print(f"  └─> Category  : [{label_name}]")
    print(f"  └─> Important : [{is_important}]")
    print(f"  └─> Reasoning : {decision.get('reasoning', "")}")

    add_labels =[]
    remove_labels =[]

    # 1. Apply the decided category label
    category_label_id = get_or_create_label(service, label_name)
    if category_label_id: 
        add_labels.append(category_label_id)

    # 2. Handle the Important system label
    if is_important:
        add_labels.append('IMPORTANT')
    else:
        # If the LLM says it's not important, we explicitly remove Gmail's native Important 
        # flag so it doesn't accidentally protect junk mail from eviction.
        remove_labels.append('IMPORTANT')

    # 3. ALWAYS apply the Processed state tag to prevent infinite loops
    processed_label_id = get_or_create_label(service, PROCESSED_LABEL)
    if processed_label_id:
        add_labels.append(processed_label_id)

    # 4. Check against the immediate archive override list
    if label_name in IMMEDIATE_ARCHIVE_LABELS and not is_important:
        remove_labels.append('INBOX')
        print(f"  └─> [Immediate Archive Triggered for '{label_name}']")
    else:
        print(f"  └─> [Kept in Inbox for Safety Period]")

    # Execute
    if DRY_RUN:
        print(f"  [DRY RUN ACTIONS] -> ADD: {add_labels} | REMOVE: {remove_labels}\n")
    else:
        service.users().messages().modify(
            userId='me', id=msg_id, body={'addLabelIds': add_labels, 'removeLabelIds': remove_labels}
        ).execute()
        print("  └─> Actions applied successfully.\n")

# ==============================================================================
# JOB RUNNERS
# ==============================================================================

def run_job_ai_sorter(service):
    """Job 1: Evaluates unprocessed inbox emails."""
    print("\n=== RUNNING JOB 1: AI INBOX SORTER ===")
    query = f"in:inbox -label:{PROCESSED_LABEL}"
    return process_batch(service, query, batch_size=50, process_function=logic_llm_triage)

def run_job_read_evictor(service):
    """Job 2: Archives processed, read emails between 30 and 60 days old. (Ignores Important)"""
    print("\n=== RUNNING JOB 2: >30-DAY-OLD & UNIMPORTANT & READ EVICTOR ===")
    # -is:important explicitly hides anything the LLM flagged as crucial
    query = f"in:inbox is:read older_than:{SAFETY_DAYS}d newer_than:{MAX_INBOX_DAYS}d label:{PROCESSED_LABEL} -is:important"
    return process_batch(service, query, batch_size=50, process_function=logic_archive_only)

def run_job_unimportant_evictor(service):
    """Job 3: Archives ALL processed emails older than 60 days. (Ignores Important)"""
    print("\n=== RUNNING JOB 3: >60-DAY-OLD & UNIMPORTANT & UNREAD EVICTOR ===")
    # -is:important explicitly hides anything the LLM flagged as crucial
    query = f"in:inbox older_than:{MAX_INBOX_DAYS}d label:{PROCESSED_LABEL} -is:important"
    return process_batch(service, query, batch_size=50, process_function=logic_archive_only)

def run_job_important_evictor(service):
    """Job 4: Archives ALL processed IMPORTANT emails older than 180 days."""
    print("\n=== RUNNING JOB 4: >180-DAY-OLD & IMPORTANT EVICTOR ===")
    # We explicitly target 'is:important' here. This catches everything 
    # (read or unread) that survived Jobs 2 & 3.
    query = f"in:inbox older_than:{MAX_IMPORTANT_DAYS}d label:{PROCESSED_LABEL} is:important"
    return process_batch(service, query, batch_size=50, process_function=logic_archive_only)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    if DRY_RUN:
        print("\n" + "="*60)
        print("🚨 DRY RUN MODE IS ENABLED 🚨")
        print("The script will read emails and evaluate them via the LLM,")
        print("but NO emails will actually be modified, archived, or labeled.")
        print("="*60)
        
    gmail_service = get_gmail_service()
    
    while True:
        a = run_job_ai_sorter(gmail_service)
        break
        b = run_job_read_evictor(gmail_service)
        c = run_job_unimportant_evictor(gmail_service)
        d = run_job_important_evictor(gmail_service) 
        if a or b or c or d:
            continue
        else:
            break

    print("\nRoutine complete.")
    if DRY_RUN:
        print("If the logic looks correct, set DRY_RUN = False to execute for real!")
