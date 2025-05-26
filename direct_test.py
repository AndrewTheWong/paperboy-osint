import os
from supabase import create_client
SUPABASE_URL = 'https://kyoowijhqqqnlmqceuka.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt5b293aWpocXFxbmxtcWNldWthIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc3MDU4MjAsImV4cCI6MjA2MzI4MTgyMH0.f7GRfEOmaSdZtvhhJFue_bhgMTILzum_ePZ-os7f_WE'
print('Testing Supabase connection...')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print('Supabase client created successfully.')
