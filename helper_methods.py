# coding=utf-8
#!/usr/bin/env python3

import logging
logging.basicConfig(format=' %(asctime)s  %(name)-12s %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__+" : ")


def clean_country_name(instring):
	name = instring.strip().upper()
	if name in ['BAHAMAS, THE']:
		return 'BAHAMAS'
	if name in [x.upper() for x in ['Bolivia (Plurinational State of)']]:
		return 'BOLIVIA'
	elif name in ['BOSNIA']:
		return 'BOSNIA AND HERZEGOVINA'
	elif name in ['BRUNEI DARUSSALAM']:
		return 'BRUNEI'
	elif name.count("IVOIRE") > 0 or name.count("COTE") > 0:
		return "COTE D'IVOIRE"
	elif name in ['CZECHIA']:
		return 'CZECH REPUBLIC'
	elif name in ['CONGO DEMOCRATIC REPUBLIC', 'CONGO DR', 'CONGO, DEM. REP.', 'Congo (Democratic Republic of the)'.upper()]:
		return 'DEMOCRATIC REPUBLIC OF THE CONGO'
	elif name in ['EGYPT, ARAB REP.']:
		return 'EGYPT'
	elif name in ['GAMBIA, THE']:
		return 'GAMBIA'
	elif name in ['GUINEA BISSAU', 'GUINEA-BISSAU']:
		return 'GUINEA-BISSAU'
	elif name.upper() in ['HONG KONG SAR, CHINA', 'HONG KONG SAR CHINA']:
		return 'HONG KONG'
	elif name in ['IRAN, ISLAMIC REP.', 'IRAN (ISLAMIC REPUBLIC OF)']:
		return 'IRAN'
	elif name in ['ISRAEL', 'ISRAEL AND WEST BANK']:
		return 'ISRAEL'
	elif name in ['KYRGYZ REPUBLIC', 'KYRGYZSTAN']:
		return 'KYRGYZSTAN'
	elif name in ['LAO PDR', 'LAO PEOPLES DEMOCRATIC REPUBLIC', 'LAOS', "Lao People's Democratic Republic".upper()]:
		return 'LAOS'
	elif name in ['MACAO SAR, CHINA']:
		return 'MACAO'
	elif name in ['MACEDONIA, FYR', 'MACEDONIA, THE FORMER YUGOSLAV REPUBLIC OF', 'THE FORMER YUGOSLAV REPUBLIC OF MACEDONIA', "Macedonia (the former Yugoslav Republic of)".upper()]:
		return 'MACEDONIA'
	elif name in ['MICRONESIA, FED. STS.', "Micronesia (Federated States of)".upper()]:
		return 'MICRONESIA'
	elif name in ['MOLDOVA', 'MOLDOVA REPUBLIC OF', 'MOLDOVA, REPUBLIC OF', "Moldova (Republic of)".upper()]:
		return 'MOLDOVA'
	elif name in ['NORTH KOREA',  'KOREA DPR', "Korea (Democratic People's Republic of)".upper()] or name.count('KOREA, DEM') > 0:
		return 'NORTH KOREA'
	elif name in ["Palestine, State of".upper()]:
		return 'PALESTINE'
	elif name in ['CONGO REPUBLIC', 'CONGO, REP.']:
		return 'REPUBLIC OF THE CONGO'
	elif name in ["Reunion !RÃ©union".upper()]:
		return 'REUNION ISLANDS'
	elif name in ['RUSSIA', 'RUSSIAN FEDERATION']:
		return 'RUSSIA'
	elif name in ['SLOVAK REPUBLIC', 'SLOVAKIA', 'SLOVAKIA (SLOVAK REPUBLIC)']:
		return 'SLOVAKIA'
	elif name in ['KOREA REPUBLIC OF', 'SOUTH KOREA', 'KOREA, REPUBLIC OF', 'KOREA', 'KOREA, REP.', 'Korea (Republic of)'.upper()]:
		return 'SOUTH KOREA'
		# return 'KOREA'
	elif name in ['SYRIAN ARAB REPUBLIC']:
		return 'SYRIA'
	elif name in ['TAIWAN', 'TAIWAN, PROVINCE OF CHINA', "Taiwan, Province of China[a]".upper()]:
		return 'TAIWAN'
		# return 'CHINA'
	elif name in ['TANZANIA', 'TANZANIA, UNITED REPUBLIC OF']:
		return 'TANZANIA'
	elif name in ['EMIRATES']:
		return 'UNITED ARAB EMIRATES'
	elif name in ['UK', "United Kingdom of Great Britain and Northern Ireland".upper()]:
		return 'UNITED KINGDOM'
	elif name in ['UNITED STATES', 'UNITED STATES OF AMERICA', 'US']:
		return 'UNITED STATES OF AMERICA'
	elif name in ['VENEZUELA, RB']:
		return 'VENEZUELA'
	elif name in ['VIET NAM']:
		return 'VIETNAM'
	elif name in ['YEMEN, REP.']:
		return 'YEMEN'
	else:
		return name
