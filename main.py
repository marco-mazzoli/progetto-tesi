import util

url_regional_data = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni-PLACEHOLDER.csv'
path = r'/Users/marcomazzoli/Documents/Projects/COVID-19/dati-regioni'

frame = read_multiple_csv(path)

region_focus = 'Emilia-Romagna'
attribute_focus = 'denominazione_regione'

region_focus_data = select_relevant_rows(frame, attribute_focus, region_focus)
region_focus_data.head()

frame_interesting_columns = select_attributes(frame, ['data', 'ricoverati_con_sintomi', 'terapia_intensiva', 'totale_ospedalizzati'])
data = series_to_supervised(frame_interesting_columns, n_in = 20, n_out = 3)
print(data)