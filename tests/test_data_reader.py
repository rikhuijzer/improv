from src.data_reader import convert_annotated_text, convert_line_message, get_filtered_messages
from src.utils import get_project_root


def test_convert_annotated_text():
    annotated = 'upgrading to Ubuntu [13.10](UbuntuVersion) from Ubuntu [13.04](UbuntuVersion)'
    expected = 'upgrading to Ubuntu 13.10 from Ubuntu 13.04'
    assert expected == convert_annotated_text(annotated)


def test_convert_line_message():
    line = ['What IRC clients are available?', 'Software Recommendation', 'False']
    message = convert_line_message(line)
    assert 'What IRC clients are available?' == message.text
    assert not message.data['training']


def test_filter_messages():
    filename = get_project_root() / 'data' / 'askubuntu' / 'askubuntu.tsv'
    train = get_filtered_messages(filename, training=True)
    assert 53 == len(train)

    test = get_filtered_messages(filename, training=False)
    assert 109 == len(test)
    assert 'Software Recommendation' == test[0].data['intent']

