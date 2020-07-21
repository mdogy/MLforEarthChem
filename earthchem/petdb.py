"""
This library is for the analysis and parsing of petdb
"""

def samples_to_dataframe(list_of_specimen_nums,petdb_url):
    """ Given a list of sample ids
    and the petdb API url, this gets all the meta-data for those sample_ids
    data is returned in "wide form" with each row being tied to a single sample,
    and single citation.
    The first two columns are specimen_num and specimen_code. 
    All other columns are any attribute that appeared for in any record. Missing values are Null.
    """
    # Needs to be implemented
    return


    