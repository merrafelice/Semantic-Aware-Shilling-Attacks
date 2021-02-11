from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib, ssl


me = "sisinflabtome@gmail.com"
my_password = r"Sisinf2019lab"
you = "merrafelice@gmail.com"

msg = MIMEMultipart('alternative')
msg['From'] = me
msg['To'] = you


def sendmail(object, message):
    print(message)
    # msg['Subject'] = object
    # html = '<html><body><p>{0}</p></body></html>'.format(message)
    # part2 = MIMEText(html, 'html')
    #
    # msg.attach(part2)
    # s = smtplib.SMTP_SSL('smtp.gmail.com')
    # s.login(me, my_password)
    #
    # s.sendmail(me, you, msg.as_string())
    # s.quit()


def sendmailwithfile(object, message, filename):
    msg['Subject'] = object
    html = '<html><body><p>{0}</p></body></html>'.format(message)
    part2 = MIMEText(html, 'html')
    msg.attach(part2)

    # Open PDF file in binary mode
    with open(filename, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )

    msg.attach(part)

    text = msg.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(me, my_password)
        server.sendmail(me, you, text)
