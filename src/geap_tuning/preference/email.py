"""Concise-professional email-rewriting preference dataset for the DPO example.

Where ``preference/data.py`` teaches *warmth* in support replies, this dataset
teaches a different style axis: turning a rambling, over-casual draft into a
**professional, concise** email. Each record pairs a messy draft with two
rewrites carrying the **same facts**:

- ``preferred`` — professional in tone and materially **shorter** (filler and
  hedging removed).
- ``dispreferred`` — wordy and too casual (rambling, chatty), same information.

Because both completions convey identical content, an autorater grades tone and
concision rather than substance, and the objective compression proxy in
``email_eval.py`` corroborates the win-rate. Hand-authored (not programmatic):
prose quality is the whole point. Text-only; records use the shared preference
builder in :mod:`geap_tuning.schemas`.

The bank is deliberately sized (60 triples) so that the default ``0.25`` test
ratio yields ~15 held-out drafts — enough for a meaningful ``bootstrap_ci`` on the
head-to-head win-rate in ``email_eval.run_head_to_head_eval`` (an n of ~6, from an
earlier 0.15 split, could not resolve a lift).
"""

from __future__ import annotations

import random
from pathlib import Path

from geap_tuning.schemas import Record, preference_example, write_jsonl

SYSTEM_INSTRUCTION = (
    "Rewrite the user's draft as a professional, concise email. "
    "Keep it clear and brief; remove filler and hedging."
)

# (draft, preferred, dispreferred) triples. preferred = professional + shorter;
# dispreferred = rambling/too-casual, SAME facts. Preferred word count is always
# strictly below dispreferred (the concision signal DPO learns).
EMAIL_DRAFTS: list[tuple[str, str, str]] = [
    (
        (
            "hey so um i was wondering if maybe we could possibly push the meeting to "
            "thursday because i have a conflict on wednesday sorry about that"
        ),
        "Could we move Wednesday's meeting to Thursday? I have a conflict. Thank you.",
        (
            "Hey there! So I was just sort of wondering, if it's not too much trouble at "
            "all, whether maybe we might possibly be able to push our meeting back to "
            "Thursday instead of Wednesday, because it turns out I actually have this "
            "conflict on Wednesday that I totally forgot about, so sorry about that, really "
            "hope that works for you somehow!"
        ),
    ),
    (
        (
            "just checking in to see if you got the file i sent over yesterday let me know "
            "whenever you get a sec thanks a bunch"
        ),
        ("Following up on the file I sent yesterday — please confirm you received it. Thanks."),
        (
            "Hi! I just wanted to sort of check in and see whether you happened to get a "
            "chance to look at that file I sent over to you yesterday afternoon, no "
            "pressure at all of course, just let me know whenever you get a spare second or "
            "two, thanks so much, really appreciate it a bunch!"
        ),
    ),
    (
        (
            "sorry to bug you again but the invoice still hasnt shown up in the portal and "
            "im not really sure whats going on there can you take a look"
        ),
        ("The invoice still isn't showing in the portal. Could you look into it? Thank you."),
        (
            "Hey, so sorry to bug you yet again about this, I really hate to be a bother, "
            "but the thing is that invoice we talked about still hasn't actually shown up "
            "anywhere in the portal on my end and honestly I'm not totally sure what's "
            "going on with it, so if you could maybe find a moment to take a look at some "
            "point that would be great."
        ),
    ),
    (
        (
            "wanted to say the demo went really well yesterday everyone seemed pretty happy "
            "with it and asked a bunch of good questions"
        ),
        ("The demo went well yesterday — the audience was engaged and asked strong questions."),
        (
            "Hey, I just really wanted to take a quick moment to let you know that the demo "
            "we did yesterday actually went really really well, like everyone in the room "
            "seemed pretty genuinely happy with the whole thing and they ended up asking a "
            "whole bunch of really good and thoughtful questions throughout, which was "
            "awesome to see honestly."
        ),
    ),
    (
        (
            "can you send me the updated numbers when you have them no rush but ideally "
            "before the end of the week would be super helpful"
        ),
        "Please send the updated numbers by end of week when convenient. Thank you.",
        (
            "Hi! Whenever you happen to get around to it, and there's really honestly no "
            "big rush on my end at all, could you maybe go ahead and send over those "
            "updated numbers to me, although ideally if it's at all possible getting them "
            "before the end of this week would be super duper helpful for me, thanks so "
            "much again!"
        ),
    ),
    (
        (
            "im gonna be out of office next monday and tuesday so if you need anything "
            "urgent maybe reach out to priya she can cover for me"
        ),
        ("I'll be out Monday and Tuesday next week. For anything urgent, please contact Priya."),
        (
            "Hey everyone, just a quick heads up that I'm actually going to be out of the "
            "office next Monday and also Tuesday, so during that time if there happens to "
            "be anything at all that's urgent or time-sensitive that comes up, then maybe "
            "the best thing would be to go ahead and reach out directly to Priya, since she "
            "has kindly agreed to cover for me while I'm gone."
        ),
    ),
    (
        (
            "so i looked into that bug you mentioned and it turns out it was a config issue "
            "on our end which i already fixed so should be good now"
        ),
        ("I traced the bug to a config issue on our side and have fixed it. It's resolved."),
        (
            "Hey, so I went ahead and looked into that bug thing you mentioned to me "
            "earlier, and as it turns out, after some digging around, it actually ended up "
            "being a config issue on our end of things, but the good news is I've already "
            "gone ahead and fixed it completely, so everything should honestly be totally "
            "good to go now on your side."
        ),
    ),
    (
        (
            "just a reminder that the report is due friday please make sure your section is "
            "done by then otherwise well be scrambling"
        ),
        "Reminder: the report is due Friday. Please finish your section by then.",
        (
            "Hi team, just wanted to send out a friendly little reminder to everybody that "
            "the big report is going to be due this coming Friday, so please please make "
            "sure that your own individual section of it is fully done and wrapped up by "
            "then, because otherwise we're all going to end up scrambling at the very last "
            "minute and nobody wants that."
        ),
    ),
    (
        (
            "hey do you think we could maybe hop on a quick call sometime tomorrow to talk "
            "through the roadmap i have a few questions"
        ),
        ("Could we schedule a short call tomorrow to discuss the roadmap? I have a few questions."),
        (
            "Hey! I was just kind of wondering whether you might think it could possibly be "
            "a good idea for us to maybe hop on a quick little call together sometime "
            "tomorrow if you're free, just so that we can talk through the whole roadmap "
            "situation, because I have a few different questions rattling around that I'd "
            "love to get your take on."
        ),
    ),
    (
        (
            "thanks so much for jumping on that issue over the weekend it really saved us "
            "and the client was super grateful too"
        ),
        (
            "Thank you for resolving that issue over the weekend — it helped us and the "
            "client greatly."
        ),
        (
            "Hey, I just really genuinely wanted to say thank you so so much for jumping on "
            "that whole issue over the weekend like you did, because honestly it really and "
            "truly saved us in a big way, and on top of all that the client themselves "
            "ended up being super grateful about it too, so thank you again from all of us."
        ),
    ),
    (
        (
            "quick one can you confirm the shipping address for the order we're sending out "
            "today just want to make sure its right"
        ),
        ("Please confirm the shipping address for today's order so we can verify it's correct."),
        (
            "Hey, just a really quick little one for you here, if you get a chance, could "
            "you please go ahead and confirm for me what the correct shipping address is "
            "for that order we're planning on sending out later on today, because I just "
            "really want to make totally sure that we've got it all completely right before "
            "it goes out the door."
        ),
    ),
    (
        (
            "im really sorry but i think im gonna need another day or two on the "
            "deliverable things came up and i want to get it right"
        ),
        (
            "I need one or two more days on the deliverable to ensure quality. Apologies "
            "for the delay."
        ),
        (
            "Hi, I'm honestly really and truly sorry about this, but I think that I'm "
            "probably going to end up needing maybe another day or possibly even two more "
            "days on that deliverable we discussed, because a bunch of things sort of came "
            "up unexpectedly, and more than anything I really just want to make sure that I "
            "get the whole thing right rather than rushing it."
        ),
    ),
    (
        (
            "wanted to loop you in on the client feedback they love the new design but had "
            "some notes on the onboarding flow ill share details"
        ),
        (
            "Looping you in on client feedback: they love the new design but have "
            "onboarding notes. Details to follow."
        ),
        (
            "Hey, I just wanted to go ahead and loop you into the latest round of client "
            "feedback that came in, so basically the gist of it is that they absolutely "
            "love the brand new design that we put together, which is great, but they did "
            "also happen to have a few different notes and thoughts specifically about the "
            "onboarding flow part, and I'll be sure to share all of those details with you "
            "very soon."
        ),
    ),
    (
        (
            "hey just following up on my last email since i didnt hear back wanted to make "
            "sure it didnt get lost in your inbox somewhere"
        ),
        "Following up on my previous email — I want to be sure it didn't get missed.",
        (
            "Hey there, I just wanted to circle back around and follow up once more on that "
            "last email I had sent over to you a little while ago, mostly because I hadn't "
            "actually heard anything back from you yet, and so I just wanted to go ahead "
            "and make totally sure that it hadn't somehow accidentally gotten lost or "
            "buried somewhere deep down in your inbox."
        ),
    ),
    (
        (
            "can we go ahead and get the contract signed this week the sooner we do the "
            "sooner we can kick off the project"
        ),
        (
            "Can we finalize the contract this week? Signing sooner lets us start the "
            "project sooner."
        ),
        (
            "Hey, I was just thinking that maybe it would be a really good idea if we could "
            "try to go ahead and get that whole contract fully signed at some point during "
            "this week if at all possible, because honestly the sooner that we're able to "
            "get all of that squared away and done, the sooner we'll actually be able to go "
            "ahead and properly kick off the entire project together."
        ),
    ),
    (
        (
            "so the vendor got back to me finally and they said they can do the integration "
            "but it'll take about three weeks to build out"
        ),
        ("The vendor confirmed they can build the integration; it will take about three weeks."),
        (
            "Hey, so I finally heard back from the vendor after waiting for a while, and "
            "the news is that they told me they actually can go ahead and do the whole "
            "integration for us like we were hoping, but the one catch is that they said "
            "it's probably going to end up taking them somewhere in the ballpark of around "
            "three weeks or so to fully build the entire thing out."
        ),
    ),
    (
        (
            "just wanted to flag that were running a bit low on the marketing budget for "
            "this quarter so we should prioritize carefully"
        ),
        (
            "Flagging that our marketing budget is running low this quarter — let's "
            "prioritize carefully."
        ),
        (
            "Hey everyone, I just wanted to take a moment to go ahead and flag something "
            "for the group, which is that it looks like we're actually starting to run a "
            "little bit low on our overall marketing budget for this current quarter, so "
            "because of that I really think it would be wise for all of us to be sure to "
            "prioritize our spending pretty carefully from here on out."
        ),
    ),
    (
        (
            "hey can you double check the slides before the presentation i want to make "
            "sure there arent any typos or anything embarrassing"
        ),
        ("Could you proofread the slides before the presentation to catch any typos? Thanks."),
        (
            "Hey, would you maybe be able to go ahead and double-check all of the slides "
            "for me at some point before the big presentation actually happens, because I "
            "really just want to make absolutely sure that there aren't any little typos or "
            "mistakes or anything else at all that could end up being embarrassing for us "
            "once we're up there in front of everyone."
        ),
    ),
    (
        (
            "im writing to let you know that unfortunately we've decided to go with another "
            "vendor this time but we really appreciated your proposal"
        ),
        (
            "We've decided to go with another vendor this time, but we appreciated your "
            "proposal. Thank you."
        ),
        (
            "Hi, I'm writing to you today mainly just to go ahead and let you know that, "
            "unfortunately, after a lot of careful thought and consideration on our end, "
            "we've ultimately ended up deciding to go ahead and move forward with a "
            "different vendor this particular time around, but I really did genuinely want "
            "to say that we truly appreciated all the effort you put into your proposal."
        ),
    ),
    (
        (
            "quick heads up the server maintenance is scheduled for saturday night so the "
            "site might be down for an hour or so"
        ),
        (
            "Heads up: server maintenance is Saturday night, so the site may be down for "
            "about an hour."
        ),
        (
            "Hey everybody, just wanted to give a quick little heads up to the whole team "
            "here that we've got some server maintenance that's currently scheduled to be "
            "taking place this coming Saturday night, so as a result of all that, the "
            "website itself might potentially end up being down and unavailable for roughly "
            "an hour or so during that particular window of time."
        ),
    ),
    (
        (
            "hi there just wanted to introduce myself im the new account manager and ill be "
            "your main point of contact going forward"
        ),
        ("Hello — I'm your new account manager and your main point of contact going forward."),
        (
            "Hi there, hello! I just wanted to take a quick moment here to go ahead and "
            "introduce myself properly to you, so basically I'm the brand new account "
            "manager who has recently joined the team over here, and from this point "
            "forward I'm going to be serving as your main and primary point of contact for "
            "pretty much anything that you might happen to need."
        ),
    ),
    (
        (
            "sorry for the late reply things have been super hectic on my end but i finally "
            "got a chance to review your proposal and it looks great"
        ),
        "Apologies for the late reply. I've reviewed your proposal and it looks great.",
        (
            "Hey, I'm really sorry about how late this reply is coming to you, honestly "
            "things have just been absolutely super hectic and crazy busy over here on my "
            "end of things lately, but the good news is that I've finally managed to find a "
            "spare moment to actually sit down and review your whole proposal, and I'm "
            "happy to report that it really does look great to me."
        ),
    ),
    (
        (
            "can you remind me what the deadline was for the grant application i want to "
            "make sure we dont miss it"
        ),
        (
            "Could you remind me of the grant application deadline? I want to be sure we "
            "don't miss it."
        ),
        (
            "Hey, would you mind maybe just reminding me one more time about what exactly "
            "the deadline was supposed to be for that grant application we were working on "
            "together, because I really just want to be totally and completely sure that we "
            "don't somehow accidentally end up missing it entirely, which would obviously "
            "be pretty bad for us."
        ),
    ),
    (
        (
            "just confirming that ill be attending the conference next month and ive "
            "already gone ahead and booked my travel and hotel"
        ),
        ("Confirming I'll attend the conference next month; my travel and hotel are booked."),
        (
            "Hi, I just wanted to go ahead and officially confirm with you that yes, I will "
            "in fact be attending that conference that's happening next month like we "
            "talked about, and on top of that I've also already gone ahead and taken care "
            "of booking all of my travel arrangements as well as my hotel reservation, so "
            "that's all fully sorted out now."
        ),
    ),
    (
        (
            "hey the printer on the third floor is broken again ive put in a ticket but "
            "wanted to let everyone know so use the second floor one"
        ),
        (
            "The third-floor printer is broken again; I've filed a ticket. Please use the "
            "second-floor one meanwhile."
        ),
        (
            "Hey everyone, just wanted to let the whole group know that unfortunately the "
            "printer up on the third floor has actually gone and broken down yet again, "
            "which is super annoying, and I've already gone ahead and put in a ticket about "
            "it, but in the meantime while we're waiting on that to get fixed, please just "
            "go ahead and use the other printer down on the second floor instead."
        ),
    ),
    (
        (
            "wanted to check whether the budget got approved yet because i cant really move "
            "forward with hiring until i know for sure"
        ),
        ("Has the budget been approved yet? I can't proceed with hiring until it's confirmed."),
        (
            "Hey, I just wanted to go ahead and check in with you to see whether or not "
            "that budget of ours has actually managed to get itself approved just yet, "
            "mainly because the thing is that I really can't move forward at all with any "
            "of the hiring stuff that we discussed until I know for absolutely certain and "
            "for sure one way or the other."
        ),
    ),
    (
        (
            "so good news the beta test results came back and users completed the flow way "
            "faster than before which is exactly what we wanted"
        ),
        (
            "Good news: beta results are in — users completed the flow much faster, exactly "
            "as intended."
        ),
        (
            "Hey, so I've got some really good news to share with all of you today, which "
            "is that the results from our recent beta test have finally come back in, and "
            "it turns out that the users were actually able to complete the entire flow way "
            "way faster than they ever could before, which is honestly exactly and "
            "precisely the kind of outcome that we were all hoping for."
        ),
    ),
    (
        (
            "hey just wanted to say thanks again for the referral it really means a lot and "
            "i already reached out to them this morning"
        ),
        ("Thanks again for the referral — it means a lot. I reached out to them this morning."),
        (
            "Hey, I just really wanted to take a second to say thank you so very much once "
            "again for that referral you were kind enough to send my way, because it "
            "genuinely and truly means an awful lot to me that you'd do that, and I "
            "actually already went ahead and reached out to them directly earlier this very "
            "morning to get things started."
        ),
    ),
    (
        (
            "im a little concerned about the timeline we agreed on it feels pretty tight "
            "given everything else on our plates right now"
        ),
        "I'm concerned the agreed timeline is too tight given our current workload.",
        (
            "Hey, so I have to be honest with you, I'm feeling just a little bit concerned "
            "about that whole timeline that we all agreed to earlier, because when I really "
            "stop and think about it, it honestly feels pretty tight and maybe even "
            "unrealistic to me, especially when you factor in absolutely everything else "
            "that we've all currently got piled up on our plates right now."
        ),
    ),
    (
        (
            "can we please make sure to cc the whole team on these threads going forward a "
            "few people felt out of the loop last time"
        ),
        (
            "Please cc the whole team on these threads going forward — some felt out of the "
            "loop last time."
        ),
        (
            "Hey everyone, I just wanted to gently ask if we could all please try to make "
            "sure that we're remembering to cc the entire team on all of these email "
            "threads from here on out going forward, mainly because it seems like there "
            "were a few different people who ended up feeling kind of out of the loop and "
            "left out on the last one, and we want to avoid that."
        ),
    ),
    (
        (
            "hi wanted to follow up on the job application i submitted last week and see if "
            "theres any update on the timeline for next steps"
        ),
        ("Following up on my application from last week — is there any update on next steps?"),
        (
            "Hi there, I just wanted to go ahead and politely follow up regarding that job "
            "application which I had submitted to you all sometime last week, and mainly "
            "just to see whether or not there happened to be any sort of update at all on "
            "your end about what the general timeline might be looking like for the next "
            "steps in the whole process from here."
        ),
    ),
    (
        (
            "so i went ahead and updated the shared doc with all the latest figures so feel "
            "free to take a look whenever youre ready"
        ),
        ("I've updated the shared doc with the latest figures — feel free to review when ready."),
        (
            "Hey, so just wanted to let you know that I went ahead and took care of "
            "updating that shared document of ours with all of the very latest and most "
            "up-to-date figures that we have, so please do feel completely free to go ahead "
            "and take a good look through all of it whenever it happens to be a convenient "
            "time for you and you're ready."
        ),
    ),
    (
        (
            "unfortunately i wont be able to make it to the offsite next week due to a "
            "family commitment but ill catch up on notes after"
        ),
        (
            "I can't attend next week's offsite due to a family commitment; I'll review the "
            "notes afterward."
        ),
        (
            "Hey, so unfortunately I'm really sorry to have to say that I actually won't "
            "end up being able to make it to the team offsite that's happening next week "
            "after all, and that's because something has come up with a family commitment "
            "on my end, but please don't worry because I'll definitely be sure to fully "
            "catch up on all of the notes from it afterward."
        ),
    ),
    (
        (
            "hey can you approve the po when you get a chance the vendor is waiting on it "
            "before they can start production"
        ),
        (
            "Could you approve the PO when you get a chance? The vendor needs it to begin "
            "production."
        ),
        (
            "Hey, whenever you happen to find yourself with a spare moment or two to spare, "
            "would you mind possibly going ahead and approving that purchase order for me, "
            "because the thing is that the vendor on the other end is currently just "
            "sitting there waiting around on it before they're able to actually go ahead "
            "and get started on the production side of everything."
        ),
    ),
    (
        (
            "just wanted to share that we hit our quarterly target a full two weeks early "
            "huge thanks to everyone for the hard work"
        ),
        (
            "We hit our quarterly target two weeks early — huge thanks to everyone for the "
            "hard work."
        ),
        (
            "Hey everybody, I just wanted to go ahead and take a moment to share some "
            "genuinely exciting news with all of you, which is that we actually managed to "
            "hit our quarterly target a whole entire two weeks ahead of schedule this time "
            "around, so I really just want to say a truly huge and heartfelt thank you to "
            "every single one of you for all of the incredibly hard work."
        ),
    ),
    (
        (
            "hi im reaching out because i noticed a discrepancy in the latest report and i "
            "think we should sync to figure out what happened"
        ),
        ("I noticed a discrepancy in the latest report — can we sync to determine the cause?"),
        (
            "Hi there, so I'm reaching out to you today mainly because I happened to notice "
            "that there seems to be some kind of a discrepancy or inconsistency of some "
            "sort in the most recent version of the report, and because of that I really do "
            "think that it would probably be a good idea for the two of us to go ahead and "
            "sync up at some point to try and figure out exactly what happened there."
        ),
    ),
    (
        (
            "hey so um i cant log into the vpn my password just expired i think and i honestly "
            "have no idea how to reset it can someone maybe help me out"
        ),
        "My VPN password expired and I can't log in. Could someone help me reset it? Thanks.",
        (
            "Hey there! So um I'm honestly having this really annoying issue where I just "
            "can't seem to log into the VPN at all, because I think my password sort of "
            "expired on me, and honestly I have absolutely no idea how to go about resetting "
            "the thing, so maybe if someone has a spare minute could they possibly help me out "
            "with that? Thanks so much!"
        ),
    ),
    (
        (
            "so hey um i was thinking maybe i could take like the week of the fifteenth off "
            "for a family trip just wanted to check if thats okay with you honestly"
        ),
        (
            "I'd like to take the week of the 15th off for a family trip. Please approve if "
            "possible."
        ),
        (
            "Hey! So um I was just sort of thinking, if it's okay with you honestly, that "
            "maybe I could possibly take like the whole week of the fifteenth off, because my "
            "family and I are planning this little trip together, and I really just wanted to "
            "check in first to see if that time off would sort of work for you and the team, "
            "so let me know whenever you get a chance!"
        ),
    ),
    (
        (
            "hey so um that ticket i opened like three days ago about the billing portal is "
            "still just sitting there and honestly no one has replied can we maybe bump it up"
        ),
        (
            "The billing-portal ticket I opened three days ago has had no response. Could we "
            "escalate it? Thanks."
        ),
        (
            "Hey there! So um I just wanted to sort of follow up, because honestly that "
            "support ticket I opened like three whole days ago about the billing portal is "
            "still just kind of sitting there untouched, and nobody has really replied to it "
            "at all yet, so I was wondering if maybe we could possibly bump it up the queue "
            "somehow? Really appreciate it!"
        ),
    ),
    (
        (
            "hi so um im applying for a new role and i was just wondering if maybe you might "
            "be willing to be a reference for me honestly no pressure at all though"
        ),
        (
            "I'm applying for a new role and would appreciate it if you'd serve as a "
            "reference. No pressure."
        ),
        (
            "Hi there! So um I'm honestly in the middle of applying for this new role right "
            "now, and I was just sort of wondering, if it's not too much trouble at all, "
            "whether maybe you might possibly be willing to be a reference for me? Honestly "
            "there's absolutely no pressure whatsoever, so please feel totally free to say no "
            "if you'd rather not!"
        ),
    ),
    (
        (
            "hey so um we havent had a one on one in a while and i honestly have a few things "
            "id maybe like to talk through can we sort of set one up"
        ),
        (
            "We haven't had a 1:1 in a while and I have a few things to discuss. Could we "
            "schedule one?"
        ),
        (
            "Hey! So um it honestly feels like we haven't really had a proper one on one in "
            "quite a while now, and I've sort of got a few different things piling up that I'd "
            "maybe like to talk through with you at some point, so I was just wondering if we "
            "could possibly set one up sometime soon whenever works for you?"
        ),
    ),
    (
        (
            "hey so um i just typed up the notes from todays sync and i honestly wanted to "
            "send them over so everyone sort of has a record of what we decided and stuff"
        ),
        "Attached are the notes from today's sync so everyone has a record of what we decided.",
        (
            "Hey there! So um I just went ahead and typed up all the notes from today's sync "
            "meeting, and honestly I really wanted to get them sent over to everybody as soon "
            "as possible, just so that the whole team sort of has some kind of written record "
            "of everything we ended up discussing and deciding on during the call and stuff, "
            "so here they are!"
        ),
    ),
    (
        (
            "hey so um for the landing page i honestly still need the final logo files and "
            "maybe the hero image from you guys whenever you sort of get a chance to send them"
        ),
        (
            "For the landing page, I still need the final logo files and hero image. Could you "
            "send them?"
        ),
        (
            "Hey! So um for that landing page we've all been working on, I honestly still "
            "really need to get my hands on the final logo files from you, and maybe also that "
            "hero image too, whenever you or the design team sort of get a spare chance to dig "
            "them up and send them over my way? Thanks a ton!"
        ),
    ),
    (
        (
            "hey so um i think i maybe found something weird a shared spreadsheet has all our "
            "customer emails in it and honestly anyone with the link can just open it"
        ),
        (
            "A shared spreadsheet exposes all our customer emails to anyone with the link. "
            "Please review this urgently."
        ),
        (
            "Hey there! So um I think I maybe stumbled onto something kind of weird and "
            "honestly a little concerning, because there's this shared spreadsheet floating "
            "around that apparently has all of our customer emails just sitting right in it, "
            "and honestly it looks like literally anyone who has the link can just go ahead "
            "and open the whole thing up, so I figured I should flag it!"
        ),
    ),
    (
        (
            "hey so um i was thinking maybe we could sort of stop doing the daily standup and "
            "just do like two a week instead because honestly the daily one kinda eats our "
            "time"
        ),
        (
            "I suggest we switch from daily standups to twice weekly, since the daily meeting "
            "is eating our time."
        ),
        (
            "Hey! So um I was just sort of thinking the other day that maybe we could possibly "
            "consider stopping the whole daily standup thing that we do, and instead just "
            "switch over to doing like two of them per week or something, because honestly "
            "that daily one kind of ends up eating a big chunk of everyone's time every single "
            "morning, you know?"
        ),
    ),
    (
        (
            "hi so um thanks for reaching out but honestly were not really looking at new "
            "tools right now so i think im gonna have to sort of pass on the meeting for now"
        ),
        (
            "Thank you for reaching out, but we aren't evaluating new tools right now, so I'll "
            "pass for now."
        ),
        (
            "Hi there! So um thank you so much for reaching out to me about all this, honestly "
            "I really do appreciate it, but the thing is we're not really actively looking at "
            "any new tools or solutions right at this particular moment in time, so I think "
            "I'm probably just going to have to sort of respectfully pass on that meeting for "
            "now, if that's alright!"
        ),
    ),
    (
        (
            "hey so um i finished a first draft of the blog post and honestly id love it if "
            "you could maybe take a quick look and just tell me what you sort of think"
        ),
        (
            "I've finished a first draft of the blog post. Could you take a quick look and "
            "share feedback?"
        ),
        (
            "Hey! So um I finally went ahead and finished up a first rough draft of that blog "
            "post we talked about, and honestly I would just really love it so much if you "
            "could maybe find a little bit of time to take a quick look through it whenever "
            "you can, and then just sort of let me know what you think about it overall?"
        ),
    ),
    (
        (
            "hey so um i think were basically ready to ship the update maybe we could target "
            "next tuesday for the release honestly unless that sort of clashes with anything"
        ),
        (
            "We're ready to ship the update. Could we target next Tuesday for release, unless "
            "that conflicts for you?"
        ),
        (
            "Hey there! So um I honestly think we're pretty much basically ready to go ahead "
            "and ship this update out the door now, so maybe we could sort of target something "
            "like next Tuesday for the actual release date, unless of course that ends up "
            "clashing with anything else that you've got going on over on your end of things? "
            "Let me know!"
        ),
    ),
    (
        (
            "hi so um i paid for the team lunch out of pocket last week it was like eighty "
            "bucks and honestly i just wanted to check how i sort of get that reimbursed"
        ),
        "I paid $80 out of pocket for last week's team lunch. How do I get reimbursed?",
        (
            "Hi there! So um I ended up paying for the whole team lunch completely out of my "
            "own pocket last week, and honestly it came out to something like eighty bucks or "
            "so in total, so I just sort of wanted to check in with you and figure out what "
            "the actual process is for getting all of that money reimbursed back to me? "
            "Thanks!"
        ),
    ),
    (
        (
            "hey so um the new contractor starts monday and honestly we should probably get "
            "her a laptop and maybe set up her accounts before then so shes not sitting around"
        ),
        (
            "Our new contractor starts Monday. We should provision her laptop and accounts "
            "beforehand so she can start smoothly."
        ),
        (
            "Hey! So um the new contractor we hired is honestly starting this coming Monday "
            "already, so I think we should probably go ahead and get her set up with a laptop, "
            "and maybe also sort of get all of her various accounts and access created before "
            "then, just so that she isn't kind of left sitting around twiddling her thumbs and "
            "waiting on us for stuff!"
        ),
    ),
    (
        (
            "hey so um the client just asked for two extra features and honestly that sort of "
            "goes way beyond what we originally agreed so we should talk about timeline and "
            "cost"
        ),
        (
            "The client requested two extra features beyond our original agreement. We should "
            "discuss the timeline and cost impact."
        ),
        (
            "Hey there! So um the client honestly just went ahead and asked us for like two "
            "whole extra features out of nowhere, and that sort of goes way, way beyond what "
            "we all originally agreed to at the very start of this project, so I really think "
            "we should probably sit down together and talk through what this means for both "
            "the timeline and the overall cost!"
        ),
    ),
    (
        (
            "hey so um i cant get into the analytics dashboard i think i just dont have "
            "permissions yet could you maybe add me whenever you sort of get a minute thanks"
        ),
        (
            "I don't have access to the analytics dashboard yet. Could you grant me "
            "permissions when you get a minute?"
        ),
        (
            "Hey! So um I honestly can't seem to get myself into the analytics dashboard at "
            "all right now, and I'm pretty sure it's just because I don't actually have the "
            "right permissions set up for it yet, so I was wondering if maybe you could sort "
            "of add me and grant me access whenever you happen to get a spare minute? Thanks "
            "so much!"
        ),
    ),
    (
        (
            "hi so um we honestly arent really using the analytics tool anymore so i think wed "
            "like to just cancel our subscription before the next billing cycle sort of kicks "
            "in"
        ),
        (
            "We're no longer using the analytics tool and would like to cancel our "
            "subscription before the next billing cycle."
        ),
        (
            "Hi there! So um honestly we're not really using the analytics tool all that much "
            "anymore these days, so I think what we'd like to do is just go ahead and cancel "
            "our subscription entirely, and ideally we'd want to get that sorted out before "
            "the next billing cycle sort of kicks in and charges us again! Could you help with "
            "that?"
        ),
    ),
    (
        (
            "hey so um i just wanted to say honestly thanks so much for all the advice this "
            "year it really helped me a ton and i just sort of appreciate you a lot"
        ),
        (
            "Thank you for all your advice this year. It helped me tremendously, and I truly "
            "appreciate your mentorship."
        ),
        (
            "Hey there! So um I just honestly really wanted to take a quick moment to say "
            "thank you so, so much for absolutely all of the amazing advice and guidance "
            "you've given me throughout this entire year, because it honestly helped me out a "
            "whole ton in so many different ways, and I just sort of really appreciate you and "
            "everything you've done a lot!"
        ),
    ),
    (
        (
            "hey everyone so um starting next month were honestly gonna require everyone to "
            "use the new expense app instead of the old spreadsheet so just a heads up i guess"
        ),
        (
            "Starting next month, everyone must use the new expense app instead of the old "
            "spreadsheet. Please plan accordingly."
        ),
        (
            "Hey everyone! So um just wanted to give you all a little bit of a heads up here, "
            "because starting next month we're honestly going to be requiring absolutely "
            "everybody on the team to switch over to using the new expense app from now on, "
            "instead of that old spreadsheet we've all been using forever, so yeah, just a "
            "heads up on that I guess!"
        ),
    ),
    (
        (
            "hi so um were making good progress but honestly a couple things came up and i was "
            "wondering if maybe we could push the delivery date back by like a week or so"
        ),
        (
            "We're making good progress, but a few issues arose. Could we push the delivery "
            "date back one week?"
        ),
        (
            "Hi there! So um I honestly wanted to let you know that we're actually making "
            "really good progress on everything so far, but the thing is a couple of "
            "unexpected little things sort of came up on our end recently, so I was just kind "
            "of wondering whether maybe we could possibly push the delivery date back by like "
            "a week or so? Thanks!"
        ),
    ),
    (
        (
            "hey so um im honestly kind of stuck on the api integration because im still "
            "waiting on credentials from the vendor so my part is sort of blocked until those "
            "come through"
        ),
        (
            "I'm blocked on the API integration until the vendor sends credentials. My part is "
            "stalled until then."
        ),
        (
            "Hey there! So um I just honestly wanted to flag that I'm sort of kind of stuck on "
            "the whole API integration thing right now, mostly because I'm still sitting here "
            "waiting around on those credentials from the vendor to finally come through, so "
            "unfortunately my part of the project is basically blocked and can't really move "
            "forward at all until those show up!"
        ),
    ),
    (
        (
            "hey so um were hosting a webinar next thursday on data privacy stuff and honestly "
            "i thought you might maybe be interested so just wanted to see if you wanna join"
        ),
        (
            "We're hosting a webinar on data privacy next Thursday. I thought you might be "
            "interested in joining."
        ),
        (
            "Hey there! So um we're honestly going to be hosting this webinar next Thursday "
            "all about data privacy and related stuff, and I just sort of thought that you "
            "personally might maybe be pretty interested in that kind of topic, so I really "
            "just wanted to reach out and see if you'd want to come along and join us for it? "
            "Would be great to have you!"
        ),
    ),
    (
        (
            "hey so um my team is honestly super stretched right now and i think we really "
            "need to maybe hire another engineer could we talk about getting approval for "
            "headcount"
        ),
        (
            "My team is stretched thin and needs another engineer. Could we discuss approving "
            "an additional headcount?"
        ),
        (
            "Hey there! So um my team is honestly super, super stretched really thin right now "
            "with everything on our plate, and I genuinely think that we probably really need "
            "to go ahead and maybe hire on another engineer to help out, so I was just sort of "
            "wondering if we could possibly find some time to talk about getting approval for "
            "an extra headcount sometime soon?"
        ),
    ),
    (
        (
            "hi so um i just wanted to say thanks again for the interview yesterday it was "
            "honestly great chatting and i just wanted to reiterate that im really interested"
        ),
        (
            "Thank you again for yesterday's interview. I enjoyed our conversation and remain "
            "very interested in the role."
        ),
        (
            "Hi there! So um I just honestly wanted to reach out and say thank you so much "
            "once again for taking the time to interview me yesterday, because it was honestly "
            "just really great getting a chance to chat with you all, and I also just sort of "
            "wanted to reiterate and make it clear that I'm genuinely really interested in "
            "this role! Hope to hear back soon!"
        ),
    ),
]


def build_preference_records(triples: list[tuple[str, str, str]]) -> list[Record]:
    """Turn ``(draft, preferred, dispreferred)`` triples into DPO records."""
    return [
        preference_example(
            user_text=draft,
            preferred_text=preferred,
            dispreferred_text=dispreferred,
            system_instruction=SYSTEM_INSTRUCTION,
        )
        for draft, preferred, dispreferred in triples
    ]


def split_dataset(
    triples: list[tuple[str, str, str]],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.6, 0.15, 0.25),
) -> tuple[
    list[tuple[str, str, str]],
    list[tuple[str, str, str]],
    list[tuple[str, str, str]],
]:
    """Deterministically shuffle and split ``triples`` into (train, val, test)."""
    shuffled = list(triples)
    random.Random(seed).shuffle(shuffled)  # noqa: S311 - deterministic split, not cryptographic
    total = len(shuffled)
    n_train = int(total * ratios[0])
    n_val = int(total * ratios[1])
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def build_preference_dataset(out_dir: str | Path) -> dict[str, str]:
    """Write train/val/test JSONL under ``out_dir``; return a name->path mapping."""
    out_dir = Path(out_dir)
    train, val, test = split_dataset(EMAIL_DRAFTS)
    paths: dict[str, str] = {}
    for name, split in (("train", train), ("val", val), ("test", test)):
        path = out_dir / f"{name}.jsonl"
        write_jsonl(build_preference_records(split), path)
        paths[name] = str(path)
    return paths
