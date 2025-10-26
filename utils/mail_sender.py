# utils/mail_sender.py
import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email(subject, body, recipients, sender_email, sender_password, attachments=None, smtp_server="smtp.gmail.com",
               smtp_port=587, html=False):
    """Send an email with the given subject and body to the list of recipients, with optional attachments and HTML support."""

    # Set up the MIME
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(recipients)
    msg['Subject'] = subject

    # Attach the body text to the email, either plain text or HTML
    if html:
        msg.attach(MIMEText(body, 'html'))
    else:
        msg.attach(MIMEText(body, 'plain'))

    # Attach any files if provided
    if attachments:
        for file in attachments:
            try:
                with open(file, "rb") as attachment:
                    # Create a MIMEBase object
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())

                # Encode the file in base64
                encoders.encode_base64(part)

                # Add header to the attachment
                part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(file)}')

                # Attach the file to the email
                msg.attach(part)

            except Exception as e:
                print(f"Failed to attach {file}: {e}")

    # Create the SMTP session
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection
        server.login(sender_email, sender_password)  # Use App Password here

        # Send the email
        server.send_message(msg)
        print(f"Email sent successfully to {', '.join(recipients)}")
    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        server.quit()
