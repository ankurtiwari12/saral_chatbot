# 🔄 Feedback-Based Regeneration Feature

## Overview

The system now supports **feedback-based answer regeneration**! After reviewing bullets and accepting/rejecting them, you can regenerate the answer to **keep what you accepted and remove/fix what you rejected**.

This creates a true **feedback loop** where your input directly improves the output.

---

## 🎯 How It Works

### **Step-by-Step Workflow**

#### **1. Generate Initial Answer**
```
Query: "Create 5 slides on encryption with 3 speaker notes per slide"
```
System generates the presentation.

#### **2. Enable Feedback UI**
- Sidebar → "📝 Feedback Settings"
- Check ☑️ "Enable Interactive Feedback"

#### **3. Review Bullets**
Below the answer, you'll see individual bullet cards:

```
┌──────────────────────────────────────┐
│ **Slide 1**                          │
│ • "Homomorphic encryption allows..." │
│                                      │
│ 📖 Source: paper.pdf (Page 3) ▼     │
│ ✅ Accept    ❌ Reject               │
└──────────────────────────────────────┘
```

#### **4. Accept/Reject Bullets**
- Click **✅ Accept** on accurate statements → Green border
- Click **❌ Reject** on incorrect statements → Red border/faded

#### **5. View Feedback Summary**
After reviewing, you'll see:
```
📊 Feedback Summary
Total Reviewed: 15
Accepted: 12 (80.0%)
Rejected: 3
```

#### **6. Generate Refined Answer**
Click the big button:
```
┌────────────────────────────────────────┐
│ ✨ Generate Refined Answer Based on    │
│    Feedback                            │
└────────────────────────────────────────┘
```

#### **7. Review Refinement Instruction**
The system creates an instruction like:

```
Based on user feedback, please refine the answer:

✅ **KEEP these points (user accepted 12 statement(s)):**
1. Definition: "Homomorphic encryption is a cryptographic technique..."
2. Explanation: "It provides a way to perform operations..."
3. Key Benefit: "Preserves data privacy while enabling..."
... and 9 more accepted statements.

❌ **REMOVE/FIX these points (user rejected 3 statement(s)):**
1. Application: "In secure computing, homomorphic encryption..."
2. Example: "The proposed approach uses ElGamal..."
3. Overview: "The proposed approach compares favorably..."

Please generate a refined answer that:
- Keeps all accepted statements
- Removes or corrects rejected statements
- Maintains the same format and structure
- Preserves equations and technical accuracy
```

#### **8. Submit Refinement**
Click **"📤 Submit This Refinement"** or scroll down and click **"🚀 Submit Query"**

#### **9. Get Refined Answer**
The system generates a new answer:
```
✨ Refined Answer (Based on Your Feedback)

🎉 Answer regenerated! Accepted points are preserved, 
rejected points have been removed/corrected. 
Review the new answer below and provide fresh feedback if needed.

[New answer with only accepted content + corrected rejected content]
```

#### **10. Review Again (Optional)**
The feedback UI resets, so you can review the new answer and iterate!

---

## 🎨 Visual Indicators

### **In Chat History**

**User Message:**
```
You: 🔄 [Feedback-based refinement]

Based on user feedback, please refine the answer:
...
```

**Assistant Response:**
```
Assistant: ✨ **[Refined based on your feedback]**

[Refined answer content...]
```

### **In Result Box**

**Header Changes:**
- Normal: "📝 Answer"
- Refined: "✨ Refined Answer (Based on Your Feedback)"

**Success Message:**
```
🎉 Answer regenerated! Accepted points are preserved, 
rejected points have been removed/corrected.
```

---

## 💡 Use Cases

### **Use Case 1: Remove Inaccurate Statements**

**Scenario:** LLM generates 10 bullets, but 2 are factually incorrect.

**Action:**
1. Accept 8 correct bullets ✅
2. Reject 2 incorrect bullets ❌
3. Generate refined answer
4. Result: New answer with only the 8 correct bullets

### **Use Case 2: Filter Out Redundant Content**

**Scenario:** Some bullets repeat the same information.

**Action:**
1. Accept unique bullets ✅
2. Reject redundant bullets ❌
3. Generate refined answer
4. Result: Cleaner, non-redundant answer

### **Use Case 3: Focus on Specific Aspects**

**Scenario:** Answer covers too many topics, you want focus.

**Action:**
1. Accept bullets on desired topic ✅
2. Reject off-topic bullets ❌
3. Generate refined answer
4. Result: Focused answer on desired topic

### **Use Case 4: Iterative Refinement**

**Scenario:** Multiple rounds of improvement.

**Action:**
1. Generate initial answer
2. Accept/reject bullets → regenerate
3. Review new answer
4. Accept/reject again → regenerate
5. Continue until satisfied

---

## 🔧 Technical Details

### **How Regeneration Works**

1. **Collect Feedback**
   - Scan all bullet_feedback in session state
   - Separate into accepted[] and rejected[] lists

2. **Create Refinement Prompt**
   - List accepted points (with preview)
   - List rejected points (with preview)
   - Add instructions to keep/remove/fix

3. **Submit as Refinement Query**
   - Uses existing refinement system
   - Treated as slide-specific refinement
   - Preserves format and structure

4. **Generate New Answer**
   - LLM processes refinement instruction
   - Keeps accepted content
   - Removes/corrects rejected content
   - Maintains original format

5. **Reset Feedback**
   - Clears bullet_feedback state
   - Allows fresh review of new answer

### **Session State Variables**

```python
st.session_state.bullet_feedback = {}
# {bullet_id: 'accepted'/'rejected'}

st.session_state.pending_feedback_refinement = "instruction text"
# Stores the refinement instruction

st.session_state.auto_submit_query = "refinement instruction"
# Auto-fills query box when set
```

### **Code Flow**

```
User clicks "Generate Refined Answer"
  ↓
collect accepted/rejected bullets
  ↓
create refinement_instruction
  ↓
store in pending_feedback_refinement
  ↓
show instruction in expander
  ↓
User clicks "Submit This Refinement"
  ↓
store in auto_submit_query
  ↓
page reloads
  ↓
query box auto-filled with instruction
  ↓
user clicks "Submit Query"
  ↓
query processed as refinement
  ↓
new answer generated
  ↓
marked as "Refined based on feedback"
  ↓
bullet_feedback cleared
  ↓
user can review new answer
```

---

## 📊 Example Session

```
═══════════════════════════════════════════════════════════
ROUND 1: Initial Generation
═══════════════════════════════════════════════════════════

You: "Create 5 slides on encryption with 3 speaker notes per slide"
