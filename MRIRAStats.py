# Import required libraries
import os
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
# User will need to pip install pandas and matplotlib

# Reading the configuration file
def read_config(directory):
    global entities_dict
    global events_dict
    current_section = ''
    entities_dict = OrderedDict()
    events_dict = {}

    # Change File Path to configuration file as needed
    with open(directory + "/annotation.conf", "r") as f_config:
        for line in f_config:

            # Will always contain headers [entities], [relations], [events], [attributes]
            # * is optional argument

            if line.startswith('[entities]'):
                current_section = 'entities'
            elif line.startswith('[relations]'):
                current_section = 'relations'
            elif line.startswith('[events]'):
                current_section = 'events'
            elif line.startswith('[attributes]'):
                current_section = 'attributes'

            elif line != '\n':
                if current_section == 'entities':
                    # Entities are stored in a nested ordered dictionary (for hierarchy elements)
                    if not(line.startswith('\t')):
                        # Insert [entity: empty dict] pair into entities_dict
                        entities_dict[line.strip('\n')] = OrderedDict()

                    elif line.startswith('\t\t'):
                        # Insert [entity: empty dict] pair into dict of *parent* element
                        # level1 is the *value/dict* of the last key entered into the entities_dict (grandparent dict)
                        level1 = entities_dict[next(reversed(entities_dict))]
                        # Insert pair within the last dict entered into the level1/grandparent dict (parent dict)
                        level1[next(reversed(level1))][line.strip('\t').strip('\n')] = OrderedDict()

                    elif line.startswith('\t'):
                        # Insert pair within the last dict entered into the entities_dict (parent dict)
                        entities_dict[next(reversed(entities_dict))][line.strip('\t').strip('\n')] = OrderedDict()




                elif current_section == 'events':

                    key, content = line.split('\t')

                    # Initialise a dictionary to hold the sub-keys and values
                    sub_dict = {}

                    # Split the content by commas to get each sub-key and value
                    pairs = content.split(', ')
                    for pair in pairs:
                        # Split each pair by the *: separator to get the sub-key and value
                        sub_key, value = pair.split(':')

                        # Split the value by | to get a list of items (allowed types for each argument)
                        value_list = value.split('|')
                        value_list = [s.strip('\n') for s in value_list]

                        # Assign the list to the sub-key in the sub-dictionary
                        sub_dict[sub_key.strip('*')] = value_list

                    # Assign the sub-dictionary to the main key in the result dictionary
                    events_dict[key] = sub_dict


    f_config.close()

    # Some change in formats from config data dictionaries above, to aid further processing:

    # Recursive function to get a non-hierarchical list (entities_list)
    # from the hierarchical nested dictionary (entities_dict) of all possible entity types
    def get_all_entities(dict1):
        global entities_list
        entities_list += list(dict1.keys())
        for k, v in dict1.items():
            if len(v) == 0:
                pass
            else:
                get_all_entities(v)
        return

    global entities_list
    entities_list = []
    get_all_entities(entities_dict) # This will populate entities_list

    # Get list of all possible event (trigger) types from dictionary
    event_trigger_list = (list(events_dict.keys()))


def stats_set_up():

    # Entities

    global entity_stats_dict
    global entities_list
    # Stats stored in each list, in same order of entities as in config file
    entity_stats_dict = {
        'Entity Category': entities_list,
        'TP': [0 for x in range(len(entities_list))],
        'FP': [0 for x in range(len(entities_list))],
        'FN': [0 for x in range(len(entities_list))],
        'Support': [0 for x in range(len(entities_list))],
        'Precision': [0 for x in range(len(entities_list))],
        'Recall': [0 for x in range(len(entities_list))],
        'F1 score': [0 for x in range(len(entities_list))]
    }

# Get Text Bound Annotations ONLY from one annotation file (parameter)
def get_text_bound_annotations(df):
    df_text_bound = df.loc[df['ID'].str.startswith('T')]
    df_text_bound = df_text_bound.copy()
    #text column = the actual text of each annotation instance
    df_text_bound[['Info2', 'text']] = df_text_bound['Info'].str.split('\t', expand=True)
    #annotation_type = entity type
    df_text_bound[['annotation_type', 'start_offset', 'end_offset']] = df_text_bound['Info2'].str.split(' ', expand=True)
    df_text_bound = df_text_bound.drop('Info', axis = 1)
    df_text_bound = df_text_bound.drop('Info2', axis = 1)
    df_text_bound = df_text_bound[['ID', 'annotation_type', 'start_offset', 'end_offset', 'text']]
    return df_text_bound




# Get Events ONLY from one annotation file (parameter)
def get_events(df):
    dfEvents = df.loc[df['ID'].str.startswith('E')]
    dfEvents = dfEvents.copy()

    # OPTION 1
    # All events in one df, 'args' column data kept in format from annotation file
    dfEvents[['df_type', 'text']] = dfEvents['Info'].str.split(':', n=1, expand = True)
    dfEvents[['trigger_ID', 'args']] = dfEvents['text'].str.split(' ', n=1, expand = True)
    dfEvents = dfEvents.drop('Info', axis = 1)
    dfEvents = dfEvents.drop('text', axis = 1)


    # OPTION 2
    # A dict of DATAFRAMES (one df for each event type)
    # Columns for each df are ID, trigger ID, and all specific argument types for THAT event type (from events_dict in read_config())
    dictEvents = {k: pd.DataFrame(columns=(['ID', 'trigger_ID'] + list(v.keys()))) for k, v in events_dict.items()}

    for index, row in dfEvents.iterrows():
        dfToUse = dictEvents[row['df_type']]

        length = dfToUse.shape[1] - 2
        list_of_none = [None] * length
        #None as default/empty
        dfToUse.loc[len(dfToUse)] = [row['ID'], row['trigger_ID']] + list_of_none

        if row['args'] != None:
            argsList = row['args'].split(' ')
            result_dict = {arg.split(':')[0]: arg.split(':')[1] for arg in argsList}
            for k, v in result_dict.items():
                dfToUse.loc[len(dfToUse)-1, k] = v


    # Return both options
    return dfEvents, dfToUse


# Determine number of matching/non-matching entity annotations between 2 annotation files (parameters)
def find_matching_entities(df_text_bound_1, df_text_bound_2):

    # Iterate through all the possible entity types, from the CONFIG file
    global entity_stats_dict
    # See stats_set_up function for structure of entity_stats_dict
    global entities_list
    n = -1
    for entity in entities_list:
        # Index of list (n) corresponds to entity type, in order of config file
        n += 1
        tp = entity_stats_dict['TP'][n]
        fp = entity_stats_dict['FP'][n]
        fn = entity_stats_dict['FN'][n]
        for index, row in df_text_bound_1.iterrows():
            # Get span where type = type
            if row['annotation_type'] == entity:
                start_offset = row['start_offset']
                end_offset = row['end_offset']
                # Check if same span AND type EXACTLY in file 2
                matching_row = df_text_bound_2[
                    (df_text_bound_2['start_offset'] == start_offset) &
                    (df_text_bound_2['end_offset'] == end_offset) &
                    (df_text_bound_2['annotation_type'] == entity)]
                if not (matching_row.empty):
                    tp += 1
                else:
                    fp += 1

        for index, row in df_text_bound_2.iterrows():
            # Get span where type = type
            if row['annotation_type'] == entity:
                start_offset = row['start_offset']
                end_offset = row['end_offset']
                # Check if same span AND type EXACTLY in file 1
                matching_row = df_text_bound_1[
                    (df_text_bound_1['start_offset'] == start_offset) &
                    (df_text_bound_1['end_offset'] == end_offset) &
                    (df_text_bound_1['annotation_type'] == entity)]
                if matching_row.empty:
                    fn += 1

        # Update stored stats for THIS entity in dataframe
        entity_stats_dict['TP'][n] = tp
        entity_stats_dict['FP'][n] = fp
        entity_stats_dict['FN'][n] = fn

# Calculate support/precision/recall/f1 score for ALL entities in stats dict
def calculate_f1_scores():

    global entities_list
    global entity_stats_dict
    n = -1
    for entity in entities_list:
        n += 1

        tp = entity_stats_dict['TP'][n]
        fp = entity_stats_dict['FP'][n]
        fn = entity_stats_dict['FN'][n]
        entity_stats_dict['Support'][n] = (tp+fn)
        try:
            precision = tp/(tp+fp)
        except:
            precision = 0
        entity_stats_dict['Precision'][n] = precision
        try:
            recall = tp/(tp+fn)
        except:
            recall = 0
        entity_stats_dict['Recall'][n] = recall
        try:
            f1score = (2*precision*recall)/(precision + recall)
        except:
            f1score = 0
        entity_stats_dict['F1 score'][n] = f1score

# Calculate and store basic inter-annotator stats (tp, fp, fn) at all levels
# for one report/text file, by calling other functions
def process_report(df_annotator_1, df_annotator_2):

    """
    INFO
    T is text bound annotations (actual parts of the text)
        Entity or event trigger (actual event text proof)
    E is events
        Each event annotation has a unique ID and is defined by type
        (e.g. Prescribing), event trigger (the text stating the INSTANCE of the event)
        and arguments.
    R is relation
        Equivalence relations = *	Equiv T1 T2 T3 where * is empty ID
    A is attribute

    """

    # Text Bound Annotations (named entities)
    df_text_bound_1 = get_text_bound_annotations(df_annotator_1)
    df_text_bound_2 = get_text_bound_annotations(df_annotator_2)
    find_matching_entities(df_text_bound_1, df_text_bound_2)

    # Events
    df_events_1_1, df_events_1_2 = get_events(df_annotator_1)
    df_events_2_1, df_events_2_2 = get_events(df_annotator_2)

# Find Weighted macro average and Micro average across all entities
def calculate_averages():

    # Weighted macro averaging

    # Entities
    w_m_f1_score = 0
    total_support = entity_stats_df['Support'].dropna().sum()
    for index, row in entity_stats_df.iterrows():
        f1_score = float(row['F1 score'])
        support = float(row['Support'])
        w_m_f1_score += (f1_score*(support/total_support))

    # Micro averaging

    #Entities

    # Precision
    total_tp = entity_stats_df['TP'].sum()
    total_fp = entity_stats_df['FP'].sum()
    precision = total_tp/(total_fp + total_tp)

    # Recall
    total_fn = entity_stats_df['FN'].sum()
    recall = total_tp/(total_fn + total_tp)

    # F1 score
    m_f1_score = (2*precision*recall)/(precision + recall)

    return w_m_f1_score, m_f1_score

# Remove misleading 0s and format dataframe, output to CSV
def write_to_csv():
    # Accesses calculated dataframes as global variables

    #Entities

    entity_display_df = entity_stats_df.round(3)
    entity_display_df = entity_display_df.replace({'Support': {0: '/'}})
    entity_display_df = entity_display_df.replace({'Precision': {0: '/'}})
    entity_display_df = entity_display_df.replace({'Recall': {0: '/'}})
    entity_display_df = entity_display_df.replace({'F1 score': {0: '/'}})
    print(entity_display_df.to_string())

    averages_dict = {
        'Weighted Macro Averaged F1 Score': w_m_f1_score,
        'Micro Averaged F1 Score': m_f1_score
    }
    averages_df = pd.DataFrame(averages_dict, index=[0])

    vertical_stack = pd.concat([entity_display_df, averages_df], axis=0)
    vertical_stack.to_csv('Inter Annotator Agreement (F1 score) by Entity.csv', index=False)

# Create bar chart of Entity Category against F1 score
def create_bar_chart():
    entity_stats_df_graph = entity_stats_df.copy()
    entity_stats_df_graph = entity_stats_df_graph[entity_stats_df_graph['F1 score'] != 0]

    # Order from highest to lowest F1 score
    sorted_data = reversed(sorted(zip(entity_stats_df_graph['F1 score'], entity_stats_df_graph['Entity Category'])))
    sorted_values, sorted_categories = zip(*sorted_data)

    # Create bar chart and save as PDF
    fig = plt.figure(figsize=(10, 15), num = '17_18')
    plt.xlabel("Entity Category")
    plt.ylabel("F1 Agreement Score")
    plt.title('F1 Agreement Score by Entity')
    plt.bar(sorted_categories, sorted_values)
    plt.xticks(rotation=45, ha='right')
    plt.savefig("F1_Agreement_Score_by_Entity.pdf", format = 'pdf')


# Main Code - Entry point

directory1 = "brat-data/12NRLS_Interannotator_Agrement_NRLS_Search2_May_October_2023"
directory2 = "brat-data/12NRLS_Interannotator_Agrement_NRLS_Search2_May_October_2023_2nd"
read_config(directory1) # Check config file is in directory 1, or change file path parameter
stats_set_up()

for file in os.listdir(directory1):
    file_whole = os.fsdecode(file)
    filename = os.path.splitext(file_whole)[0]

    # Only process for each new report (text file) in the directory
    if file_whole.endswith(".txt"):
        # Get annotation files for THIS report from both annotators
        df_annotator_1 = pd.read_csv((directory1+ "/" + filename+".ann"), sep='^([^\s]*)\s', engine='python', header=None).drop(0, axis=1)
        df_annotator_1.rename(columns={1: 'ID', 2: 'Info'}, inplace=True)
        df_annotator_2 = pd.read_csv((directory2+ "/" + filename+".ann"), sep='^([^\s]*)\s', engine='python', header=None).drop(0, axis=1)
        df_annotator_2.rename(columns={1: 'ID', 2: 'Info'}, inplace=True)
        # Calculate basic inter-annotator stats at all levels for THIS report/text file
        process_report(df_annotator_1, df_annotator_2)


calculate_f1_scores()
global entity_stats_dict
entity_stats_df = pd.DataFrame(entity_stats_dict)
w_m_f1_score, m_f1_score = calculate_averages()
write_to_csv()
create_bar_chart()