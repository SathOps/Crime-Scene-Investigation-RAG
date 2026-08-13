import os
from pathlib import Path

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Delete existing files in sample_data to avoid duplicate or outdated files
for existing_file in SAMPLE_DATA_DIR.glob("*.txt"):
    try:
        existing_file.unlink()
        print(f"Removed old sample file: {existing_file.name}")
    except Exception as e:
        print(f"Could not remove {existing_file.name}: {e}")

docs = {
    "01_FIR_State_vs_Ramesh.txt": """FIRST INFORMATION REPORT (FIR)
Under Section 154 Cr.P.C. / Section 173 BNSS

1. District: Pune City
2. Police Station: Shivajinagar Police Station
3. FIR No.: 142/2024
4. Date and Time of FIR: 14th November 2024, 01:15 HRS
5. Date and Time of Incident: 13th November 2024, between 21:15 HRS and 22:00 HRS
6. Place of Occurrence: Office #402, 4th Floor, Apex Towers, Shivajinagar, Pune

7. Complainant Details:
   Name: Mr. Ankit Sharma (Age 28)
   Occupation: Night Security Guard, Apex Towers
   Address: Room 12, Police Line Quarters, Shivajinagar, Pune

8. Victim Details:
   Name: Mr. Vikram Malhotra (Age 45)
   Occupation: Managing Director, Apex Logistics Pvt. Ltd.
   Status: Deceased (Shot at scene)

9. Suspect / Accused Details:
   Name: Ramesh Kumar (Age 34)
   Occupation: Former Financial Manager, Apex Logistics Pvt. Ltd.
   Address: Flat 302, Green Acres Apartment, Kothrud, Pune

10. Acts & Sections Invoked:
    Indian Penal Code (IPC) Sections 302 (Murder), 392 (Robbery), and 120B (Criminal Conspiracy)

11. Brief Facts of the Case:
    On 13th November 2024 at approximately 21:50 HRS, complainant Ankit Sharma while conducting security routine patrol on the 4th floor of Apex Towers noticed that the wooden main entrance door of Office #402 (Apex Logistics Pvt. Ltd.) was unlocked and partially open. Upon entering the premises, complainant found Managing Director Mr. Vikram Malhotra lying face down on the office floor next to his executive desk in a pool of blood with a bullet wound in his chest.

    Complainant immediately alerted police control room at 22:05 HRS. Physical examination at the scene revealed a metallic wristwatch with blood stains (Evidence Item-C) near the victim's left hand and a spent 9mm shell casing (Evidence Item-A) on the carpet floor near the executive desk. A black leather corporate briefcase (Evidence Item-E) containing financial audit ledger files was missing from Mr. Malhotra's desk.

    Complainant Ankit Sharma stated in his initial report that at approximately 21:10 HRS he observed former employee Ramesh Kumar entering the building via the rear service gate wearing a dark navy blue hooded sweatshirt. Complainant further stated he saw Ramesh Kumar hurriedly exiting via the same rear emergency exit at approximately 21:40 HRS carrying a black briefcase.

    Investigating Officer: Inspector Rajesh Deshmukh, Shivajinagar PS.
""",

    "02_Witness_Statements.txt": """LEGAL CASE WITNESS STATEMENTS
Case Reference: State of Maharashtra vs. Ramesh Kumar (FIR No. 142/2024, Shivajinagar PS)

--------------------------------------------------------------------------------
STATEMENT 01: WITNESS ANKIT SHARMA
--------------------------------------------------------------------------------
Date of Recording: 14th November 2024, 04:30 HRS
Witness Name: Mr. Ankit Sharma (Age 28)
Role: Night Security Guard, Apex Towers
Location of Recording: Shivajinagar Police Station

Statement:
"I have been working as night security guard at Apex Towers for 2 years. On 13th November 2024, my shift started at 20:00 HRS. At around 20:30 HRS, Mr. Vikram Malhotra arrived in his car and went up to his 4th floor office #402. At approximately 21:10 HRS, I was stationed near the rear service entrance gate. I saw a man walking in rapidly through the service gate. I recognized him as Mr. Ramesh Kumar, who used to work as an accountant at Apex Logistics until he was fired two weeks ago. He was wearing a dark navy blue hooded sweatshirt with the hood pulled over his head, dark jeans, and black sneakers.

At around 21:25 HRS, while I was in the ground floor lobby, I heard a loud muffled bang followed by a heavy thud coming from the upper floors, but I initially assumed it was construction work from the adjacent plot. At 21:40 HRS, while checking the perimeter, I saw Ramesh Kumar running out of the rear emergency exit door holding a black leather briefcase tightly under his right arm. He ran toward the side lane where a motorcycle was parked and sped away. When I went up to Office #402 at 21:50 HRS, I found Mr. Vikram Malhotra lying dead on the floor."

--------------------------------------------------------------------------------
STATEMENT 02: WITNESS PRIYA NAIR
--------------------------------------------------------------------------------
Date of Recording: 14th November 2024, 11:00 HRS
Witness Name: Ms. Priya Nair (Age 31)
Role: Senior Accountant, Apex Logistics Pvt. Ltd.
Location of Recording: Shivajinagar Police Station

Statement:
"I am the Senior Accountant at Apex Logistics. I worked closely with Mr. Ramesh Kumar before he was terminated on 30th October 2024. Mr. Vikram Malhotra had discovered financial irregularities amounting to INR 45 Lakhs in the company ledger managed by Ramesh. Mr. Malhotra confronted Ramesh on 30th October, fired him immediately, and gave him until 15th November to return the embezzled money or face formal criminal prosecution. Ramesh was extremely furious and threatened Mr. Malhotra during that meeting.

On the night of 13th November 2024, I left office early at around 21:00 HRS. However, while waiting across the street near the bus stop at 21:15 HRS, I claim I saw Ramesh Kumar standing outside the front main entrance of Apex Towers. Crucially, I observed him wearing a bright red leather jacket and dark blue jeans, standing near a red motorcycle. He appeared to be talking on his mobile phone for about 5 minutes before I boarded my bus at 21:20 HRS. I did not see him enter the building from the front entrance."

--------------------------------------------------------------------------------
STATEMENT 03: WITNESS SURESH VERMA
--------------------------------------------------------------------------------
Date of Recording: 15th November 2024, 10:15 HRS
Witness Name: Mr. Suresh Verma (Age 52)
Role: Building Facilities Supervisor, Apex Towers

Statement:
"I oversee maintenance at Apex Towers. On 13th November 2024, Mr. Vikram Malhotra informed me in the afternoon that he would stay late in Office #402 to complete a financial audit of company books. Regarding the CCTV system, Camera 03 monitoring the 4th floor corridor had a loose power cable and went offline at 21:12 HRS, coming back online only at 21:45 HRS.

I also confirm that on 10th November 2024 (three days before the murder), Ramesh Kumar came to the building lobby at 17:00 HRS demanding to meet Mr. Malhotra. When Mr. Malhotra refused, Ramesh shouted in front of staff: 'You will pay for destroying my career and reputation!'"
""",

    "03_CCTV_Analysis_Report.txt": """DIGITAL FORENSIC & CCTV ANALYSIS REPORT
Department: Cyber Crime & Digital Forensics Unit, Pune City Police
Requisition Ref: FIR No. 142/2024 (Shivajinagar PS)
Report Date: 17th November 2024
Examined Asset: Hard Drive DVR Unit (Western Digital 2TB, Serial #WD-9941X) — Evidence Item-D
Seized From: Security Control Room, Apex Towers, Shivajinagar, Pune

EXAMINATION FINDINGS & TIMELINE RECONSTRUCTION:

1. Camera 01 — Rear Service Entrance & Emergency Stairwell Exit:
   - Timestamp 21:08:12 HRS (13-Nov-2024):
     A male individual matching the height (approx. 5ft 10in) and physical build of suspect Ramesh Kumar is captured entering through the rear service gate. The individual is clad in a dark navy blue hooded sweatshirt with the hood pulled over his head, dark trousers, and black footwear.
   - Timestamp 21:39:45 HRS (13-Nov-2024):
     The same male individual in the dark navy blue hooded sweatshirt is captured hurriedly exiting the rear service emergency stairwell door. He is carrying a black rectangular leather briefcase (matching missing Evidence Item-E) in his right hand.

2. Camera 02 — Main Front Entrance Gate:
   - Timestamp 21:14:50 HRS (13-Nov-2024):
     Witness Ms. Priya Nair is recorded exiting the main glass doors of Apex Towers and walking toward the street bus stop.
   - Discrepancy Verification (Priya Nair Statement):
     Detailed frame-by-frame examination of Camera 02 footage between 21:00 HRS and 21:30 HRS shows NO person wearing a red leather jacket standing outside or near the front gate. The claim made by Witness Priya Nair regarding observing Ramesh Kumar in a bright red jacket at the front entrance at 21:15 HRS is factually unsupported by video evidence.

3. Camera 03 — 4th Floor Corridor (Outside Office #402):
   - Timestamp 21:12:00 HRS to 21:45:00 HRS (13-Nov-2024):
     Video stream registered a power drop signal at 21:12:00 HRS, resulting in black video recording until manual cable reconnect at 21:45:00 HRS. Forensic log verification indicates physical disconnection of power cord at 4th floor junction box.

4. Biometric Gait Analysis & Comparison:
   - Gait parameters extracted from Camera 01 (rear service gate, 21:08:12 HRS) were compared against baseline reference video of Ramesh Kumar recorded during police custody.
   - Result: Structural Gait Analysis yielded an 87.4% positive match score (Stride Length: 74cm, Cadence: 112 steps/min, Left arm swing angle deviation).

Report Compiled By: Dr. A. K. Deshpande, Senior Forensic Scientist, Cyber Forensics Lab, Pune.
""",

    "04_Forensic_Lab_Report.txt": """STATE FORENSIC SCIENCE LABORATORY (SFSL) REPORT
Regional Center: Ganeshkhind, Pune, Maharashtra
Report No.: SFSL/BALLISTICS/2024/7821
Date of Issue: 18th November 2024
Case Reference: FIR No. 142/2024, Shivajinagar PS (State vs. Ramesh Kumar)

EVIDENCE ITEMS SUBMITTED FOR ANALYSIS:
- Evidence Item-A: 9mm Spent Brass Shell Casing recovered from Office #402 floor.
- Evidence Item-B: Plaster Cast Impression & Digital Photo of Bloody Footprint on Office #402 tiles.
- Evidence Item-C: Silver Metallic Titan Chrono Wristwatch (Model T-804) with blood smears.
- Evidence Item-F: Latent Fingerprint Lifts (FP-8821) from Office #402 executive desk and door handle.
- Evidence Item-G: Dermal Nitrate GSR Swab taken from suspect Ramesh Kumar on 14-Nov-2024 at 07:00 HRS.
- Item-H: Pair of Black Chevron-soled Sneakers (Size 9) seized from Ramesh Kumar's residence.

LABORATORY FINDINGS:

1. Ballistics Examination (Item-A):
   - Item-A is a standard 9mm Parabellum spent brass cartridge case bearing firing pin impression 'FPI-9M'.
   - Microscopic comparison against 9mm bullet recovered from victim's thoracic cavity during autopsy confirms both bullet and casing were fired from the same semi-automatic firearm. Firing pin striation characteristics indicate a 9mm pistol weapon.

2. Fingerprint Analysis (Dactyloscopy - Item-F):
   - Latent fingerprint FP-8821 lifted from the right drawer handle of the executive desk in Office #402 was developed using black magnetic powder.
   - Automated Fingerprint Identification System (AFIS) database comparison with ten-print card of Ramesh Kumar revealed 12 identical ridge characteristics (bifurcations, ridge endings, and dots) matching the right index finger of Ramesh Kumar.

3. Footwear Impression Analysis (Item-B vs. Item-H):
   - The bloody footwear impression (Item-B) recovered from the scene exhibits a Size 9 (European 43) Chevron wave sole pattern with distinctive heel wear on the outer lateral edge.
   - Comparative examination against black sneakers (Item-H) seized from Ramesh Kumar's residence confirmed 100% morphological match in tread pattern, dimensions, and individual wear marks.

4. Gunshot Residue (GSR) Test (Item-G):
   - Atomic Absorption Spectroscopy (AAS) analysis of dermal swabs (Item-G) taken from Ramesh Kumar's right palm and index finger tested POSITIVE for elevated levels of Lead (Pb), Barium (Ba), and Antimony (Sb).
   - Conclusion: The chemical profile is characteristic of primer residue resulting from firing a firearm within 12 to 18 hours prior to sample collection.

5. Serology & DNA Profiling (Item-C):
   - Blood sample extracted from Titan Wristwatch (Item-C) yielded a single-source DNA profile matching victim Vikram Malhotra (Probability of match: 1 in 4.2 Billion).

Chief Forensic Examiner: Dr. S. R. Thorne, Director of Forensic Ballistics & Fingerprints, SFSL Pune.
""",

    "05_Postmortem_Report.txt": """POSTMORTEM EXAMINATION REPORT
Department of Forensic Medicine & Toxicology
Sassoon General Hospital & B.J. Medical College, Pune
Autopsy PM Report No.: PM-894/2024
Date & Time of Autopsy: 14th November 2024, 08:30 HRS to 11:15 HRS

DECEASED PARTICULARS:
- Name: Vikram Malhotra
- Age / Gender: 45 Years / Male
- Height / Weight: 175 cm / 78 kg
- Identified By: Mr. Suresh Verma (Supervisor) & Police Constable Constable Patil (Buckle #4412)
- Body Brought By: Inspector Rajesh Deshmukh, Shivajinagar PS

EXTERNAL INJURIES:
1. Gunshot Entry Wound:
   A circular wound measuring 9mm in diameter located on the left anterior chest wall, 4 cm below the left nipple and 2 cm lateral to the sternum. Margin of wound is inverted. Thermal tattooing, soot deposition, and skin singeing present in a 3cm radius surrounding the entry wound, indicating firearm discharge at close range (estimated firing distance under 2 feet / 60 cm).

2. Gunshot Exit Wound:
   An irregular, lacerated exit wound measuring 14mm x 11mm located on the posterior thoracic region at the level of the 7th intercostal space. Margins are everted with profuse tissue damage.

3. Minor Contusions:
   A blunt-force abrasion measuring 2cm x 1cm on the knuckles of the right hand, consistent with defensive impact or struggle prior to collapse.

INTERNAL EXAMINATION & ANATOMICAL FINDINGS:
- Respiratory System: Trachea contains blood aspiration. Left lung perforated through upper lobe.
- Cardiovascular System: Massive laceration of the anterior wall of the left ventricle of the heart and thoracic aorta. Approximately 1,800 ml of fluid and clotted blood present in the pericardial and pleural cavities.
- Gastrointestinal System: Stomach contains approximately 150ml of semi-digested rice and vegetable curry. State of gastric digestion indicates meal was ingested approximately 2 to 3 hours prior to death.

ESTIMATED TIME OF DEATH:
- Rigor mortis: Fully established in facial muscles, neck, and upper limbs; partially established in lower extremities.
- Postmortem Lividity: Present on posterior aspects of trunk and thighs, fixed.
- Medical Opinion on Time of Death: Based on rigor mortis progression, postmortem lividity, and gastric digestion state, death occurred between 10.5 and 11.5 hours prior to autopsy examination.
- Conclusive Time Window: Estimated Time of Death is fixed between 21:15 HRS and 21:35 HRS on 13th November 2024.

CAUSE OF DEATH:
"Death caused by Hypovolemic Shock and Internal Hemorrhage resulting from a close-range firearm injury to the heart (Gunshot Wound to Chest)."

Autopsy Surgeon: Dr. Manisha Kulkarni, MD (Forensic Medicine), Sassoon Hospital, Pune.
""",

    "06_Police_Investigation_Report.txt": """POLICE INVESTIGATION & CHARGE SHEET SUMMARY
Office of the Assistant Commissioner of Police, Shivajinagar Division, Pune City
Charge Sheet No.: CS/142/2024
Date of Filing: 1st December 2024
Investigating Officer: Inspector Rajesh Deshmukh, Senior PI, Shivajinagar PS

SUMMARY OF INVESTIGATION & EVIDENCE SYNTHESIS:

1. Establishment of Motive:
   Investigating agency established that accused Ramesh Kumar was employed as Senior Accountant at Apex Logistics Pvt. Ltd. Internal audit revealed that Ramesh had systematically embezzled INR 45 Lakhs over 14 months into shell bank accounts under his brother-in-law's name. Deceased MD Vikram Malhotra discovered the fraud on 30th October 2024, terminated Ramesh, and demanded full restitution by 15th November 2024 under threat of police filing. This provided direct malice and motive for murder.

2. Timeline & Reconstructing the Crime (13th November 2024):
   - 20:30 HRS: Victim Vikram Malhotra arrives at Office #402.
   - 21:04 HRS: Accused Ramesh Kumar arrives in Shivajinagar area. Cell Tower CDR logs show his mobile number 98230-77123 connected to Tower Shivajinagar-BS-04.
   - 21:08 HRS: Ramesh enters Apex Towers via rear service gate in a navy hoodie (CCTV Cam 01).
   - 21:12 HRS: Ramesh disables 4th floor corridor CCTV power cable.
   - 21:15 - 21:30 HRS: Ramesh enters Office #402, confrontations ensue. Ramesh shoots Malhotra in chest at close range with an unlicensed 9mm pistol, steals the briefcase containing audit ledgers (Evidence Item-E).
   - 21:39 HRS: Ramesh exits rear gate carrying stolen briefcase (CCTV Cam 01).
   - 21:48 HRS: Ramesh leaves Shivajinagar cell tower zone.

3. Call Detail Records (CDR) & Cell Tower Dump Analysis:
   - Handset IMEI #358912094410210 / Mobile No. 98230-77123 belonging to Ramesh Kumar registered active cell tower handovers:
     - 20:40 HRS: Kothrud Cell Tower (Residence)
     - 21:04 HRS to 21:48 HRS: Shivajinagar-BS-04 (Apex Towers vicinity)
     - 22:15 HRS: Kothrud Cell Tower (Residence)
   - This CDR record completely disproves Ramesh Kumar's claim of being in Aundh at Blue Dart Cafe (which is serviced by Aundh-BS-12 cell tower).

4. Resolution of Witness Statement Contradictions:
   - Witness Priya Nair's claim of seeing Ramesh outside front gate at 21:15 HRS in a bright red leather jacket was investigated. CCTV Camera 02 footage verified no person in a red jacket was present. Tower CDR and Guard Ankit Sharma's statement confirmed Ramesh entered via rear gate at 21:08 HRS wearing a navy hoodie. Nair's observation was determined to be a factual error or mistaken identity.

5. Final Charge: Accused Ramesh Kumar charged under Sections 302 (Murder) and 392 (Robbery) IPC.

Submitted By: Inspector Rajesh Deshmukh, PI Shivajinagar PS.
""",

    "07_Accused_Statement.txt": """STATEMENT OF THE ACCUSED (RECORD OF EXAMINATION / INTERROGATION)
Recorded under Section 313 Cr.P.C. / Section 351 BNSS
Court of the Judicial Magistrate First Class (JMFC), Court No. 3, Pune
Case Ref: State vs. Ramesh Kumar (FIR No. 142/2024)
Date of Recording: 16th November 2024
Deponent: Ramesh Kumar, S/o Suresh Kumar, Age 34, R/o Kothrud, Pune.

EXAMINATION & QUESTIONS BY COURT / INVESTIGATING OFFICER:

Q1: You have been charged with the murder of Mr. Vikram Malhotra on 13th November 2024 at Office #402, Apex Towers. What do you have to say?
Answer: "I am completely innocent. I did not kill Mr. Vikram Malhotra. I was nowhere near Shivajinagar or Apex Towers on the evening of 13th November 2024. This is a false case orchestrated by Mr. Malhotra's business rivals and family."

Q2: Where were you on the night of 13th November 2024 between 20:30 HRS and 22:30 HRS?
Answer (Alibi Claim): "On 13th November 2024, from 20:45 HRS to 22:15 HRS, I was at 'Blue Dart Cafe' located in Aundh, Pune, having coffee and discussing personal matters with my cousin Rajesh Kumar. I reached the cafe at 20:45 HRS and stayed until 22:15 HRS. I never visited Apex Towers at any time on 13th November."

Q3: Evidence shows your fingerprints (FP-8821) on the desk drawer of Office #402 and your bloody footprint (Size 9 Chevron pattern) at the scene. How do you explain this?
Answer: "I worked at Apex Logistics as an accountant until 30th October 2024. My fingerprints on the desk drawer are from my previous working days before I was terminated. As for the footprint, Chevron pattern sneakers are very common in the market, thousands of people wear them."

Q4: Forensic analysis recovered Gunshot Residue (GSR) on your right hand, and ballistics link the 9mm casing to a pistol. Did you possess or fire a weapon?
Answer: "I have never owned, held, or fired any gun or firearm in my entire life. I don't know how chemical residue came onto my hands. Perhaps the police contaminated my hands during my arrest on 14th November morning."

Q5: Cell tower records (CDR) show your mobile phone (98230-77123) active in Shivajinagar near Apex Towers from 21:04 HRS to 21:48 HRS on 13th November. Why were you in Shivajinagar if you claim you were in Aundh?
Answer: "On that evening, I left my mobile phone in the car of an acquaintance named Rahul who drove to Shivajinagar. I did not have my mobile phone with me while I was at Blue Dart Cafe in Aundh."

Q6: Do you have anything further to add regarding the alleged embezzlement of INR 45 Lakhs?
Answer: "Mr. Vikram Malhotra framed me for accounting discrepancies to cover up his own tax fraud. I did not steal any money or briefcase."

Recorded in open court, read over to the deponent and admitted to be correct.
Signature of Accused: Ramesh Kumar
Signature of Magistrate: JMFC Court No. 3, Pune.
""",

    "08_Court_Judgment.txt": """IN THE SESSIONS COURT OF PUNE, AT PUNE, MAHARASHTRA
SESSIONS CASE NO. 312 OF 2024
(Arising out of FIR No. 142/2024, Shivajinagar Police Station)

State of Maharashtra  ... Prosecution
       Versus
Ramesh Kumar          ... Accused / Defendant

Presiding Judge: Hon'ble Sessions Judge M. V. Kulkarni, B.Sc., LL.M.
Date of Pronouncement of Judgment: 10th January 2025

JUDGMENT & FINAL ORDER:

1. Prosecution Case Summary:
   The prosecution alleged that accused Ramesh Kumar murdered victim Vikram Malhotra, MD of Apex Logistics Pvt. Ltd., on 13th November 2024 between 21:15 HRS and 21:35 HRS in Office #402, Apex Towers, Shivajinagar, Pune, by firing a 9mm pistol at close range, and robbed a briefcase containing financial audit ledgers to suppress evidence of his INR 45 Lakhs embezzlement.

2. Evaluation of Defense Alibi:
   The accused pleaded an alibi under Section 11 of the Indian Evidence Act, claiming presence at Blue Dart Cafe in Aundh from 20:45 HRS to 22:15 HRS on the night of the crime.
   Court Finding: The defense of alibi is rejected in toto. Prosecution produced Cell Tower CDR evidence (Shivajinagar-BS-04) proving accused's phone was active at Shivajinagar during the crime window. Further, CCTV gait analysis (87.4% match) and testimony of Blue Dart Cafe manager confirmed accused was NOT present at the cafe.

3. Evaluation of Witness Statements & Contradictions:
   Court examined the contradiction between Security Guard Ankit Sharma (stating accused entered rear gate at 21:08 HRS in a navy blue hoodie) and Senior Accountant Priya Nair (stating she saw accused at front gate at 21:15 HRS in a red jacket).
   Court Finding: CCTV Camera 01 and Camera 02 conclusive video logs corroborate Security Guard Ankit Sharma's timeline and hoodie description. Priya Nair's observation at front gate was unsupported by video evidence and constitutes an honest error in timing or identity. Minor discrepancy does not shake the core prosecution narrative.

4. Scientific & Forensic Evidence:
   - Gunshot Residue (GSR) test confirmed Lead, Barium, and Antimony on accused's right hand.
   - Footwear impression Item-B matched Chevron sole sneakers (Item-H) seized from accused's house.
   - Fingerprint FP-8821 on victim's desk drawer matched accused's right index finger (12 ridge points).
   - Postmortem report PM-894/2024 established Time of Death between 21:15 HRS and 21:35 HRS, perfectly aligning with CCTV exit timestamp (21:39 HRS).

5. FINAL VERDICT & SENTENCE:
   The Court holds that the prosecution has established the chain of circumstantial and scientific evidence beyond all reasonable doubt.

   ORDER:
   i. Accused Ramesh Kumar is hereby CONVICTED of the offence punishable under IPC Section 302 (Murder) and IPC Section 392 (Robbery).
   ii. For offence under IPC Section 302, Accused Ramesh Kumar is sentenced to IMPRISONMENT FOR LIFE and to pay a fine of INR 50,000.
   iii. For offence under IPC Section 392, Accused Ramesh Kumar is sentenced to RIGOROUS IMPRISONMENT FOR 7 YEARS and fine of INR 10,000.
   iv. Both sentences to run concurrently.

Pronounced in Open Court on this 10th day of January 2025.
Signed: (M. V. Kulkarni) Sessions Judge, Pune.
"""
}

for filename, content in docs.items():
    file_path = SAMPLE_DATA_DIR / filename
    file_path.write_text(content.strip(), encoding="utf-8")
    print(f"Successfully generated: {filename} ({len(content)} chars)")

print("All 8 synthetic case documents created successfully!")
