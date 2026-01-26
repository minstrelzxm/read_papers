import openreview
import time

def count_notes(client, venue_id):
    try:
        notes = client.get_all_notes(content={'venueid': venue_id})
        print(f"Venue: {venue_id} -> {len(notes)} papers")
        return len(notes)
    except Exception as e:
        print(f"Venue: {venue_id} -> Error: {e}")
        return 0

def check_venues():
    client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
    
    venues_to_check = [
        'NeurIPS.cc/2025/Conference',
        'NeurIPS.cc/2025/Conference/Submission', # Sometimes accepted papers are still here but marked
    ]
    
    # Also check via invitation
    print("Checking specific invitations...")
    try:
        submissions = client.get_all_notes(invitation='NeurIPS.cc/2025/Conference/-/Submission', details='content')
        print(f"Invitation 'NeurIPS.cc/2025/Conference/-/Submission' -> {len(submissions)} total submissions")
        
        # Filter accepted
        accepted = [n for n in submissions if 'venueid' in n.content and 'NeurIPS.cc/2025' in n.content['venueid']['value']]
        print(f"   -> {len(accepted)} have venueid set (Accepted)")
    except Exception as e:
        print(f"Error checking invitation: {e}")

    for v in venues_to_check:
        count_notes(client, v)

if __name__ == "__main__":
    check_venues()
