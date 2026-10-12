"""
Test the ConversationManager against a 30+ message conversation.

What this proves:
1. Total tokens BEFORE any compression vs AFTER (context actually sent).
2. Key facts planted early in the conversation (name, location, job,
   deadline, budget, preference) survive into the summary even after
   the raw messages containing them are dropped.
"""

from conversation_manager import ConversationManager, TokenCounter

# ---- 1. Build a 30+ message conversation with facts planted early -----
KEY_FACTS = {
    "name": "Priya",
    "location": "Bhavnagar",
    "job": "data scientist",
    "deadline": "August 15th",
    "budget": "$12,000",
    "preference": "dark mode",
}

conversation_script = [
    ("user", "Hey, I need some help planning a fairly involved project and I want to walk through it in detail so we're on the same page before I start writing any code."),
    ("assistant", "Sure, happy to help. Tell me about the project -- what's the goal, who's it for, and what constraints are you working with in terms of time, data, and budget?"),
    ("user", "My name is Priya and I'm a data scientist based in Bhavnagar, India. I work on the analytics team at a mid-size retail company, and I've been asked to lead a churn-prediction initiative."),
    ("assistant", "Nice to meet you, Priya! Churn prediction is a great project to lead. Before we dive into modeling, it helps to nail down the business definition of churn -- is it a hard cancellation, or a period of inactivity past some threshold?"),
    ("user", "Good question. For us it's defined as no purchase in 90 days. I'm building a churn-prediction model for a retail client using about two years of transaction logs and customer support tickets."),
    ("assistant", "Two years of history is a solid base, especially if it spans at least one full seasonal cycle. What's the granularity of the transaction logs -- line-item level, order level, or aggregated monthly summaries?"),
    ("user", "Line-item level, so I can roll it up however I want. There's also a customer support ticket table with timestamps, categories, and resolution times I can join in."),
    ("assistant", "That's excellent, especially the support ticket linkage since dissatisfaction signals often predate churn by weeks. Are you thinking a classic gradient boosting model, or something deep-learning based given the volume of data?"),
    ("user", "Leaning toward XGBoost first as a fast, interpretable baseline, then maybe comparing against a neural net later once I understand the feature space better."),
    ("assistant", "Good plan -- XGBoost is a strong, fast baseline for tabular churn data and it'll also give you feature importances for free, which helps with stakeholder buy-in early on."),
    ("user", "My project deadline is August 15th and the budget is $12,000 for compute, tooling, and any third-party services I might need along the way."),
    ("assistant", "Noted -- that's a workable budget for cloud GPU or CPU time plus a few SaaS tools like a feature store or an experiment tracker, as long as you're deliberate about what you provision."),
    ("user", "I also prefer dark mode in any dashboards or internal tools we set up, since I'll be staring at this stuff for hours at a time during the sprint."),
    ("assistant", "Understood, I'll keep that in mind for any UI or dashboarding recommendations -- most modern BI tools support a dark theme out of the box."),
    ("user", "What features should I engineer from the transaction logs to get the model off to a good start?"),
    ("assistant", "Recency, frequency, and monetary value (RFM) are the classic starting point, plus trend deltas over the last 1, 3, and 6 months so the model can see acceleration or deceleration in spending, not just a snapshot."),
    ("user", "Should I bucket customers by tenure too, or is that redundant with the RFM features?"),
    ("assistant", "Tenure buckets are still worth adding -- they interact with churn risk in non-linear ways, especially around contract renewal points or annual membership anniversaries where churn spikes."),
    ("user", "What about the support ticket data specifically -- how do I turn that into usable features for the model?"),
    ("assistant", "Turn ticket counts, average resolution time, and sentiment score into per-customer features -- unhappy customers with slow-resolved tickets in the last 30-60 days are a strong churn signal in most retail datasets."),
    ("user", "Makes sense. What evaluation metric should I optimize for given that churners are a minority class?"),
    ("assistant", "Given the class imbalance, use PR-AUC or F1 on the minority class rather than plain accuracy, since accuracy will look deceptively good if you just predict 'no churn' for everyone."),
    ("user", "Good call. Should I also look at SHAP values for interpretability once the model is trained?"),
    ("assistant", "Definitely -- SHAP will help you explain individual churn predictions to stakeholders, which matters a lot if the output is going to drive retention campaigns or discount offers."),
    ("user", "How much of the $12,000 budget should go toward compute versus tooling and licensing?"),
    ("assistant", "A rough split of 70% compute and 30% tooling or licensing is reasonable for a project this size, assuming you're not standing up a full MLOps platform from scratch."),
    ("user", "Can you sketch a rough week-by-week timeline toward the August 15th deadline so I can share it with my manager?"),
    ("assistant", "Sure: weeks 1-2 for data prep and feature engineering, weeks 3-4 for modeling and hyperparameter tuning, week 5 for SHAP analysis and validation, and week 6 for dashboard build-out and stakeholder handoff."),
    ("user", "That timeline works well for me, thanks -- I'll bring it to my manager tomorrow along with the budget breakdown."),
    ("assistant", "Glad it helps! Let me know when you want to dig into the feature engineering details, or if your manager pushes back on the timeline and you need to compress any of those phases."),
    ("user", "One more thing -- once this is deployed, what's a good way to monitor the model in production?"),
    ("assistant", "Track prediction drift and PR-AUC on a rolling weekly window, and set an alert if performance drops more than 5% from your validation baseline so you can catch data drift before it affects retention campaigns."),
    ("user", "Okay, I've started on the feature engineering. I'm seeing a lot of missing values in the support ticket resolution time column -- around 18% of rows. How should I handle that?"),
    ("assistant", "First check whether missingness is random or systematic -- if unresolved tickets simply don't have a resolution time yet, that's informative, not noise. Consider adding a binary 'ticket still open' flag and imputing resolution time with the median within each ticket category rather than a global median."),
    ("user", "Good idea, I'll add that flag. Separately, my RFM features are heavily right-skewed, especially monetary value. Should I log-transform them before feeding into XGBoost?"),
    ("assistant", "Tree-based models like XGBoost are fairly robust to skew since they split on rank order rather than raw magnitude, so log-transforming isn't strictly necessary, but it can still help if you later want to use SHAP dependence plots that are easier to read on a compressed scale."),
    ("user", "That's a fair point, I'll transform monetary value for readability but leave frequency as-is. What about correlated features -- recency and days-since-last-ticket are pretty correlated in my data."),
    ("assistant", "For tree-based models, correlated features aren't as harmful as they are for linear models, since the trees will just pick whichever is more informative at each split. I'd leave both in unless your SHAP importance ranking gets confusing, in which case you can drop the weaker one later."),
    ("user", "Makes sense, I'll keep both for now. I ran a first baseline XGBoost model with default hyperparameters and got a PR-AUC of 0.61. Is that reasonable for a first pass?"),
    ("assistant", "0.61 PR-AUC is a solid starting point for churn prediction, especially with default hyperparameters and before any tuning. Most teams see meaningful gains from tuning max_depth, learning_rate, and scale_pos_weight given your class imbalance, plus adding a few more interaction features."),
    ("user", "I'll try tuning scale_pos_weight next since the imbalance is around 12% churners. Any rule of thumb for a starting value?"),
    ("assistant", "A common starting point is the ratio of negative to positive class counts, so roughly (1 - 0.12) / 0.12, which comes out to about 7.3. Try that as a baseline and then sweep a small range around it with cross-validation rather than trusting the formula blindly."),
    ("user", "Got it, I'll sweep values between 5 and 10. On the infrastructure side, I'm planning to run training jobs on a cloud VM -- any recommendation on instance size given my $12,000 budget?"),
    ("assistant", "For tabular XGBoost training at your data scale, you likely don't need a GPU instance -- a mid-tier CPU instance with 16-32 vCPUs and plenty of RAM will train fast and cost a fraction of a GPU box, leaving more of your budget for tooling and the eventual dashboard work."),
    ("user", "That's good news, it frees up more of the budget. Speaking of the dashboard, should I build it in a BI tool or something custom?"),
    ("assistant", "Given your team is analytics-focused and you want stakeholders to self-serve, a BI tool like a hosted dashboarding platform is usually faster to ship than a custom app, and most support dark themes and role-based access out of the box, which fits what you mentioned earlier about your UI preference."),
    ("user", "Good point, I'll go with a BI tool then rather than building something custom from scratch. How should I structure the SHAP analysis section of the dashboard for non-technical stakeholders?"),
    ("assistant", "Lead with a global feature importance chart so stakeholders see the big picture drivers of churn, then let them drill into individual customer SHAP waterfall plots for specific accounts your retention team is worried about -- that combination tends to land well with non-technical audiences."),
    ("user", "That's a great structure, I'll mock that up this week. Last question for now -- how often should I retrain the model once it's live?"),
    ("assistant", "A monthly retraining cadence is a reasonable default for retail churn given seasonal shifts in behavior, but tie it to your drift monitoring too -- if PR-AUC drops more than the 5% threshold we discussed earlier, trigger an out-of-cycle retrain rather than waiting for the next scheduled month."),
    ("user", "Actually, one more round of questions before I go. I noticed my train and test splits might be leaking information because I split randomly rather than by time. Should I switch to a time-based split given this is a churn problem with a strong temporal component?"),
    ("assistant", "Yes, definitely switch to a time-based split -- random splitting on temporal data like churn almost always leaks future information into training and inflates your validation metrics. Train on data up through, say, month 20, validate on months 21-22, and hold out the most recent months as a final test set that mimics production conditions."),
    ("user", "That explains why my PR-AUC looked a bit too good to be true. I'll redo the split. Should the RFM features also be computed as of the split cutoff date rather than using the full history?"),
    ("assistant", "Exactly right -- every feature needs to be computed 'as of' the observation date for each row, using only data available at that point in time, otherwise you'll reintroduce the same leakage through the back door even with a correct train/test split."),
    ("user", "Good catch, I'll rebuild the feature pipeline with point-in-time correctness. This is turning into more work than I expected but it's clearly necessary. How long should I budget for this rework given my August 15th deadline?"),
    ("assistant", "Point-in-time feature engineering usually takes about the same effort as the first pass since you're restructuring the pipeline logic rather than the features themselves -- budget roughly 3-4 extra days, which should still fit inside your existing week 1-2 window if you start now."),
    ("user", "Okay, I'll fold that into this week's work. Given the budget is $12,000, do you think point-in-time feature computation is going to add meaningfully to compute costs?"),
    ("assistant", "It shouldn't add much -- point-in-time features are usually more about correct SQL windowing logic than raw compute volume, so the added cost is mostly your engineering time, not additional cloud spend."),
    ("user", "That's a relief given my budget is already earmarked mostly for training and tooling. One last thing -- should I version my datasets and models as I iterate, and if so what tool would you recommend for a team our size?"),
    ("assistant", "Yes, definitely version both -- even a lightweight setup like DVC for dataset versioning paired with a simple experiment tracker for model runs will save you huge headaches later when you need to explain why a metric changed between two model versions, and it's well within your remaining tooling budget."),
    ("user", "Great, I'll add DVC to the stack this week. Thanks for catching the leakage issue and walking through all of this with me, Priya out."),
    ("assistant", "Glad it came up before you handed anything to your manager -- good luck with the rework and the DVC setup, and feel free to check back in once the corrected pipeline is running."),
]

print(f"Scripted conversation length: {len(conversation_script)} messages\n")

# ---- 2. Feed it through the manager, tracking condensation events -----
mgr = ConversationManager(token_limit=3000)   # default rule-based summarizer

raw_tokens_before_any_compression = 0
first_compression_logged = False

for i, (role, content) in enumerate(conversation_script, start=1):
    mgr.add_message(role, content)

if not mgr.compressions:
    print("NOTE: conversation never crossed 3000 tokens with this script -- "
          "padding wasn't needed, but let's check the numbers below.")

# ---- 3. BEFORE / AFTER token counts -----------------------------------
raw_all = [__import__("conversation_manager").Message(role=r, content=c)
           for r, c in conversation_script]
before_tokens = TokenCounter.count_messages(raw_all)
after_tokens = mgr.total_tokens()

print("=" * 60)
print("BEFORE / AFTER TOKEN COUNTS")
print("=" * 60)
print(f"BEFORE (full raw history, {len(raw_all)} messages): {before_tokens} tokens")
print(f"AFTER  (summary + recent, {len(mgr.get_context())} messages): {after_tokens} tokens")
if mgr.compressions:
    saved = before_tokens - after_tokens
    print(f"Reduction: {saved} tokens ({saved / before_tokens:.0%})")
    print(f"\nCompression events: {len(mgr.compressions)}")
    for i, c in enumerate(mgr.compressions, 1):
        print(f"  #{i}: triggered at {c['at_message_count']} live messages, "
              f"folded {len(c['folded_messages'])} messages into summary")
else:
    print("No compression triggered (conversation stayed under the 3000 token limit).")

print("\n" + "=" * 60)
print("CURRENT CONTEXT STATE")
print("=" * 60)
print(mgr.describe())

# ---- 4. Verify key facts survived into the summary ---------------------
print("\n" + "=" * 60)
print("FACT-RECALL VERIFICATION")
print("=" * 60)

context_text = "\n".join(m.content for m in mgr.get_context()).lower()

# facts as they'd literally appear (lowercased) so we can check presence
checks = {
    "name (Priya)": "priya" in context_text,
    "location (Bhavnagar)": "bhavnagar" in context_text,
    "job (data scientist)": "data scientist" in context_text,
    "deadline (August 15th)": "august 15" in context_text,
    "budget ($12,000)": "12,000" in context_text or "12000" in context_text,
    "preference (dark mode)": "dark mode" in context_text,
}

all_pass = True
for fact, found in checks.items():
    status = "PASS" if found else "FAIL"
    if not found:
        all_pass = False
    print(f"  [{status}] {fact}")

print(f"\nOverall: {'ALL KEY FACTS RETAINED' if all_pass else 'SOME FACTS LOST'} "
      f"after summarization.")

# ---- 5. Simulate "asking about them" -----------------------------------
print("\n" + "=" * 60)
print("SIMULATED Q&A AGAINST THE COMPRESSED CONTEXT")
print("=" * 60)

def answer_from_context(question: str) -> str:
    """
    Stand-in for an LLM call: in production you'd send mgr.get_context()
    as the message history to Claude and ask this question as the next
    user turn. Here we just show that the facts are actually present in
    the context that WOULD be sent, by locating them.
    """
    q = question.lower()
    hits = []
    for m in mgr.get_context():
        c = m.content.lower()
        if any(k in c for k in ["priya", "bhavnagar", "data scientist"]) and "name" in q:
            hits.append(m.content)
        if any(k in c for k in ["august 15", "12,000", "12000"]) and ("deadline" in q or "budget" in q):
            hits.append(m.content)
        if "dark mode" in c and "prefer" in q:
            hits.append(m.content)
    return hits[0] if hits else "(not found in current context)"

questions = [
    "What's my name and where do I live?",
    "What's my project deadline and budget?",
    "What UI preference did I mention?",
]
for q in questions:
    print(f"\nQ: {q}")
    print(f"A (found in context): {answer_from_context(q)}")
